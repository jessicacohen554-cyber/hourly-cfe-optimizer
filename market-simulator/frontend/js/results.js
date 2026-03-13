/**
 * Market Simulator — Results Page
 * Renders Plotly.js charts from simulation results stored in sessionStorage.
 */

const UNIT_COLORS = {
    coal_steam: '#374151',
    gas_ccgt: '#6B7280',
    gas_ccgt_efficient: '#4B5563',
    gas_ccgt_old: '#9CA3AF',
    gas_ct: '#D1D5DB',
    oil_ct: '#92400E',
};

const CLEAN_COLORS = {
    solar: '#F59E0B',
    wind: '#22C55E',
    offshore_wind: '#009688',
    nuclear: '#6366F1',
    ccs_ccgt: '#64748B',
    hydro: '#0EA5E9',
    battery: '#06B6D4',
    ldes: '#E91E63',
    geothermal: '#D97706',
};

const PLOTLY_LAYOUT_BASE = {
    font: { family: 'DM Sans, sans-serif' },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    margin: { t: 30, r: 30, b: 50, l: 60 },
    autosize: true,
};

// ── Load data ──
let simResult = null;
let simParams = null;
let currentISO = null;

function init() {
    const raw = sessionStorage.getItem('simulationResult');
    const rawParams = sessionStorage.getItem('simulationParams');

    if (!raw) {
        showNoData();
        return;
    }

    simResult = JSON.parse(raw);
    simParams = rawParams ? JSON.parse(rawParams) : {};

    // Handle multi-ISO results (sweep or multi-ISO run)
    if (Array.isArray(simResult)) {
        currentISO = simResult[0]?.iso || 'ERCOT';
        buildISOSelector(simResult.map(r => r.iso));
    } else if (simResult.results && typeof simResult.results === 'object') {
        // Keyed by ISO
        const isos = Object.keys(simResult.results);
        currentISO = isos[0];
        buildISOSelector(isos);
    } else {
        currentISO = simResult.iso || 'ERCOT';
        buildISOSelector([currentISO]);
    }

    renderAll();
}

function getISOResult(iso) {
    if (Array.isArray(simResult)) {
        return simResult.find(r => r.iso === iso) || simResult[0];
    }
    if (simResult.results) {
        return simResult.results[iso] || Object.values(simResult.results)[0];
    }
    return simResult;
}

function showNoData() {
    document.querySelector('.content-section').innerHTML = `
        <div class="no-data">
            <h3>No simulation results</h3>
            <p>Run a simulation from the <a href="/">setup page</a> first.</p>
        </div>`;
}

function buildISOSelector(isos) {
    const container = document.getElementById('resultsIsoSelector');
    container.innerHTML = isos.map(iso =>
        `<button class="iso-btn ${iso === currentISO ? 'active' : ''}" data-iso="${iso}">${iso}</button>`
    ).join('');

    container.querySelectorAll('.iso-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            container.querySelectorAll('.iso-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentISO = btn.dataset.iso;
            renderAll();
        });
    });
}

function renderAll() {
    const data = getISOResult(currentISO);
    if (!data) return;

    const iso = data.iso || currentISO;
    document.getElementById('resultsSubtitle').textContent =
        `${iso} — ${data.mode || 'snapshot'} simulation`;

    updateStats(data);
    renderMeritOrder(data);
    renderProfitability(data);
    renderNuclearRevenue(data);
    renderLMPImpact(data);
    renderCCSBreakeven(data);
    renderWhatBuilt(data);
    renderFossilBins(data);
    renderCostLadder(data);
    renderGasShift(data);
    renderSensitivity(data);
    renderGeneratorTable(data);
}

// ── Stats ──
function updateStats(data) {
    setText('statCleanPct', fmtPct(data.market_outcome_clean_pct || data.existing_clean_pct));
    setText('statAvgLMP', '$' + fmtNum(data.avg_lmp || data.lmp_summary?.avg));
    setText('statNewGW', fmtNum((data.new_capacity_gw || 0), 1) + ' GW');
    setText('statNucRev', '$' + fmtNum(data.nuclear_revenue?.total_mwh));
    setText('statCCSBreakeven', '$' + fmtNum(data.ccs_breakeven_carbon_price) + '/ton');
}

// ── Chart 1: Merit-Order Stack ──
function renderMeritOrder(data) {
    const gens = data.generator_economics || data.merit_order_stack || [];
    if (!gens.length) { placeholderChart('chartMeritOrder'); return; }

    const sorted = [...gens].sort((a, b) => a.marginal_cost - b.marginal_cost);
    const traces = [{
        x: sorted.map(g => g.unit_type.replace('_', ' ')),
        y: sorted.map(g => g.capacity_mw / 1000),
        marker: { color: sorted.map(g => UNIT_COLORS[g.unit_type] || '#9CA3AF') },
        type: 'bar',
        name: 'Capacity (GW)',
        hovertemplate: '%{x}<br>%{y:.1f} GW<br>MC: $%{customdata:.1f}/MWh<extra></extra>',
        customdata: sorted.map(g => g.marginal_cost),
    }];

    const avgLMP = data.avg_lmp || data.lmp_summary?.avg || 0;
    const layout = {
        ...PLOTLY_LAYOUT_BASE,
        yaxis: { title: 'Capacity (GW)', gridcolor: '#f0f0f0' },
        xaxis: { title: '' },
        shapes: [{
            type: 'line', x0: -0.5, x1: sorted.length - 0.5,
            y0: 0, y1: 0, yref: 'paper',
            line: { color: '#EF4444', width: 0 }
        }],
        annotations: [{
            x: sorted.length - 1, y: avgLMP / (Math.max(...sorted.map(g => g.capacity_mw/1000)) || 1) * 0.5,
            text: `Avg LMP: $${avgLMP.toFixed(1)}`,
            showarrow: false, font: { color: '#EF4444', size: 12 }
        }],
    };

    Plotly.newPlot('chartMeritOrder', traces, layout, { responsive: true });
}

// ── Chart 2: Generator Profitability ──
function renderProfitability(data) {
    const gens = data.generator_economics || [];
    if (!gens.length) { placeholderChart('chartProfitability'); return; }

    const names = gens.map(g => g.unit_type.replace(/_/g, ' '));
    const revenue = gens.map(g => g.avg_revenue_mwh || 0);
    const cost = gens.map(g => g.marginal_cost || 0);
    const profit = gens.map(g => g.profit_mwh || 0);

    const traces = [
        { x: names, y: revenue, name: 'Revenue $/MWh', type: 'bar', marker: { color: '#22C55E' } },
        { x: names, y: cost, name: 'Cost $/MWh', type: 'bar', marker: { color: '#EF4444' } },
        { x: names, y: profit, name: 'Profit $/MWh', type: 'bar', marker: { color: profit.map(p => p >= 0 ? '#6366F1' : '#F59E0B') } },
    ];

    Plotly.newPlot('chartProfitability', traces, {
        ...PLOTLY_LAYOUT_BASE,
        barmode: 'group',
        yaxis: { title: '$/MWh', gridcolor: '#f0f0f0' },
    }, { responsive: true });
}

// ── Chart 3: Nuclear Revenue ──
function renderNuclearRevenue(data) {
    const nuc = data.nuclear_revenue;
    if (!nuc) { placeholderChart('chartNuclearRevenue'); return; }

    const categories = ['Energy', 'Capacity', 'PTC', 'Total'];
    const values = [nuc.energy_rev_mwh, nuc.capacity_rev_mwh, nuc.ptc_mwh, nuc.total_mwh];
    const colors = ['#6366F1', '#0EA5E9', '#22C55E', '#1e293b'];

    Plotly.newPlot('chartNuclearRevenue', [{
        x: categories, y: values, type: 'bar',
        marker: { color: colors },
        hovertemplate: '%{x}: $%{y:.1f}/MWh<extra></extra>',
    }], {
        ...PLOTLY_LAYOUT_BASE,
        yaxis: { title: '$/MWh', gridcolor: '#f0f0f0' },
    }, { responsive: true });
}

// ── Chart 4: LMP Impact ──
function renderLMPImpact(data) {
    const sweep = data.threshold_sweep || data.lmp_by_threshold;
    if (!sweep) { placeholderChart('chartLMPImpact'); return; }

    const thresholds = Object.keys(sweep).map(Number).sort((a, b) => a - b);
    const lmps = thresholds.map(t => sweep[t]?.avg_lmp || sweep[t]);

    Plotly.newPlot('chartLMPImpact', [{
        x: thresholds, y: lmps, type: 'scatter', mode: 'lines+markers',
        line: { color: '#6366F1', width: 3 },
        marker: { size: 6 },
        hovertemplate: '%{x}% clean: $%{y:.1f}/MWh<extra></extra>',
    }], {
        ...PLOTLY_LAYOUT_BASE,
        xaxis: { title: 'Clean Energy %', gridcolor: '#f0f0f0' },
        yaxis: { title: 'Avg LMP ($/MWh)', gridcolor: '#f0f0f0' },
    }, { responsive: true });
}

// ── Chart 5: CCS Breakeven ──
function renderCCSBreakeven(data) {
    const ccs = data.ccs_analysis || data.ccs_breakeven;
    if (!ccs) { placeholderChart('chartCCSBreakeven'); return; }

    const carbonPrices = ccs.carbon_prices || [0, 25, 50, 75, 100, 125, 150, 175, 200];
    const existingCost = ccs.existing_ccgt_cost || carbonPrices.map(cp => 30 + cp * 0.37);
    const ccsCost = ccs.ccs_retrofit_cost || carbonPrices.map(cp => 65 + cp * 0.037);
    const newGasCost = ccs.new_gas_cost || carbonPrices.map(cp => 35 + cp * 0.37);

    Plotly.newPlot('chartCCSBreakeven', [
        { x: carbonPrices, y: existingCost, name: 'Existing CCGT', mode: 'lines', line: { color: '#6B7280', width: 3 } },
        { x: carbonPrices, y: ccsCost, name: 'CCS Retrofit', mode: 'lines', line: { color: '#64748B', width: 3, dash: 'dash' } },
        { x: carbonPrices, y: newGasCost, name: 'New Efficient Gas', mode: 'lines', line: { color: '#22C55E', width: 3 } },
    ], {
        ...PLOTLY_LAYOUT_BASE,
        xaxis: { title: 'Carbon Price ($/ton)', gridcolor: '#f0f0f0' },
        yaxis: { title: '$/MWh Total Cost', gridcolor: '#f0f0f0' },
    }, { responsive: true });
}

// ── Chart 6: What Gets Built ──
function renderWhatBuilt(data) {
    const built = data.what_gets_built || data.deployment_stack;
    if (!built) { placeholderChart('chartWhatBuilt'); return; }

    const resources = Object.keys(built);
    const values = resources.map(r => built[r]);
    const colors = resources.map(r => CLEAN_COLORS[r] || '#9CA3AF');

    Plotly.newPlot('chartWhatBuilt', [{
        labels: resources.map(r => r.replace(/_/g, ' ')),
        values: values,
        type: 'pie',
        marker: { colors: colors },
        textinfo: 'label+percent',
        hovertemplate: '%{label}: %{value:.1f} GW (%{percent})<extra></extra>',
    }], {
        ...PLOTLY_LAYOUT_BASE,
    }, { responsive: true });
}

// ── Chart 7: Fossil Bin Economics ──
function renderFossilBins(data) {
    const bins = data.fossil_bin_economics || data.generator_economics;
    if (!bins || !bins.length) { placeholderChart('chartFossilBins'); return; }

    const names = bins.map(b => b.unit_type.replace(/_/g, ' '));

    Plotly.newPlot('chartFossilBins', [
        { x: names, y: bins.map(b => b.dispatch_hours || 0), name: 'Dispatch Hours', type: 'bar', yaxis: 'y', marker: { color: '#6366F1' } },
        { x: names, y: bins.map(b => (b.capacity_factor || 0) * 100), name: 'CF %', type: 'scatter', mode: 'lines+markers', yaxis: 'y2', line: { color: '#EF4444', width: 3 }, marker: { size: 8 } },
    ], {
        ...PLOTLY_LAYOUT_BASE,
        yaxis: { title: 'Dispatch Hours', gridcolor: '#f0f0f0' },
        yaxis2: { title: 'Capacity Factor %', overlaying: 'y', side: 'right', range: [0, 100] },
        barmode: 'group',
    }, { responsive: true });
}

// ── Chart 8: Cost Ladder ──
function renderCostLadder(data) {
    const ladder = data.cost_ladder;
    if (!ladder) { placeholderChart('chartCostLadder'); return; }

    const cumGW = ladder.map(s => s.cumulative_gw);
    const costs = ladder.map(s => s.cost_mwh);
    const revenues = ladder.map(s => s.revenue_mwh);

    Plotly.newPlot('chartCostLadder', [
        { x: cumGW, y: costs, name: 'Cost $/MWh', type: 'scatter', mode: 'lines+markers', line: { color: '#EF4444', width: 3 } },
        { x: cumGW, y: revenues, name: 'Revenue $/MWh', type: 'scatter', mode: 'lines+markers', line: { color: '#22C55E', width: 3 } },
    ], {
        ...PLOTLY_LAYOUT_BASE,
        xaxis: { title: 'Cumulative New Clean GW', gridcolor: '#f0f0f0' },
        yaxis: { title: '$/MWh', gridcolor: '#f0f0f0' },
    }, { responsive: true });
}

// ── Chart 9: Gas Shift ──
function renderGasShift(data) {
    const shift = data.gas_fleet_shift || data.fleet_shift;
    if (!shift) { placeholderChart('chartGasShift'); return; }

    // Expect array of {carbon_price, efficient_cf, avg_cf, old_cf}
    if (Array.isArray(shift)) {
        Plotly.newPlot('chartGasShift', [
            { x: shift.map(s => s.carbon_price), y: shift.map(s => s.efficient_cf * 100), name: 'Efficient CCGT', mode: 'lines', line: { color: '#22C55E', width: 3 } },
            { x: shift.map(s => s.carbon_price), y: shift.map(s => s.avg_cf * 100), name: 'Average CCGT', mode: 'lines', line: { color: '#6B7280', width: 3 } },
            { x: shift.map(s => s.carbon_price), y: shift.map(s => s.old_cf * 100), name: 'Old CCGT', mode: 'lines', line: { color: '#9CA3AF', width: 3, dash: 'dash' } },
        ], {
            ...PLOTLY_LAYOUT_BASE,
            xaxis: { title: 'Carbon Price ($/ton)', gridcolor: '#f0f0f0' },
            yaxis: { title: 'Capacity Factor %', gridcolor: '#f0f0f0', range: [0, 100] },
        }, { responsive: true });
    } else {
        placeholderChart('chartGasShift');
    }
}

// ── Chart 10: Sensitivity Heatmap ──
function renderSensitivity(data) {
    const sens = data.sensitivity_matrix;
    if (!sens) { placeholderChart('chartSensitivity'); return; }

    Plotly.newPlot('chartSensitivity', [{
        z: sens.values,
        x: sens.gas_prices,
        y: sens.carbon_prices,
        type: 'heatmap',
        colorscale: 'Viridis',
        colorbar: { title: 'Clean %' },
        hovertemplate: 'Gas: $%{x}/MMBtu<br>Carbon: $%{y}/ton<br>Clean: %{z:.1f}%<extra></extra>',
    }], {
        ...PLOTLY_LAYOUT_BASE,
        xaxis: { title: 'Gas Price ($/MMBtu)' },
        yaxis: { title: 'Carbon Price ($/ton)' },
    }, { responsive: true });
}

// ── Generator Table ──
function renderGeneratorTable(data) {
    const gens = data.generator_economics || [];
    const tbody = document.getElementById('generatorTableBody');
    if (!tbody) return;

    tbody.innerHTML = gens.map(g => `
        <tr>
            <td>${g.unit_type.replace(/_/g, ' ')}</td>
            <td>${fmtNum(g.capacity_mw, 0)}</td>
            <td>$${fmtNum(g.marginal_cost)}</td>
            <td>${fmtNum(g.dispatch_hours, 0)}</td>
            <td>${fmtPct(g.capacity_factor * 100)}</td>
            <td>$${fmtNum(g.avg_revenue_mwh)}</td>
            <td style="color: ${g.profit_mwh >= 0 ? '#166534' : '#991b1b'}">$${fmtNum(g.profit_mwh)}</td>
            <td><span class="status-badge status-${g.status || 'marginal'}">${g.status || 'unknown'}</span></td>
        </tr>
    `).join('');
}

// ── Helpers ──
function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text || '--';
}

function fmtNum(val, decimals = 1) {
    if (val == null || isNaN(val)) return '--';
    return Number(val).toFixed(decimals);
}

function fmtPct(val) {
    if (val == null || isNaN(val)) return '--';
    return Number(val).toFixed(1) + '%';
}

function placeholderChart(containerId) {
    const el = document.getElementById(containerId);
    if (el) {
        el.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#9CA3AF;font-style:italic;">No data for this chart</div>';
    }
}

// ── Collapsible sections ──
document.querySelectorAll('.collapsible-header').forEach(header => {
    header.addEventListener('click', () => {
        const target = document.getElementById(header.dataset.target);
        if (target) {
            const isOpen = target.style.maxHeight && target.style.maxHeight !== '0px';
            target.style.maxHeight = isOpen ? '0px' : '2000px';
            header.querySelector('.collapse-icon').style.transform = isOpen ? '' : 'rotate(180deg)';
        }
    });
});

// ── Initialize ──
init();
