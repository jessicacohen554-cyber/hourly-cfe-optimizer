/**
 * Market Simulator — Results Page
 * Renders Plotly.js charts from simulation results stored in sessionStorage.
 *
 * Layout hierarchy:
 *   Level 1: Market-wide (LMP time series, supply stack, fuel bin table, merit order)
 *   Level 2: Year drill-down (profitability, deployment, CCS, gas shift)
 *   Level 3: IPP fleet analysis (generator economics table)
 */

const UNIT_COLORS = {
    coal_steam: '#2C3E50',
    coal_steam_efficient: '#34495E',
    coal_steam_avg: '#2C3E50',
    coal_steam_old: '#1A252F',
    gas_ccgt: '#2372B9',
    gas_ccgt_efficient: '#2372B9',
    gas_ccgt_avg: '#1B5E9E',
    gas_ccgt_old: '#144A7D',
    gas_ct: '#007FA4',
    gas_ct_efficient: '#007FA4',
    gas_ct_avg: '#006A8A',
    gas_ct_old: '#005570',
    oil_ct: '#9B6B3A',
};

const CLEAN_COLORS = {
    solar: '#FBB254',
    wind: '#6BA543',
    offshore_wind: '#007FA4',
    nuclear: '#2372B9',
    ccs_ccgt: '#7F8F97',
    hydro: '#007FA4',
    battery: '#CADB2E',
    ldes: '#F47B27',
    geothermal: '#9B6B3A',
};

// Combined resource colors (clean + fossil)
const ALL_RESOURCE_COLORS = {
    ...CLEAN_COLORS,
    coal_steam: '#2C3E50',
    gas_ccgt: '#2372B9',
    gas_ct: '#007FA4',
    oil_ct: '#9B6B3A',
    fossil: '#6B7280',
};

const PLOTLY_LAYOUT_BASE = {
    font: { family: "'Source Sans Pro', Calibri, Arial, sans-serif", color: '#1A232F' },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    margin: { t: 40, r: 30, b: 50, l: 60 },
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

    // Determine current ISO
    if (Array.isArray(simResult)) {
        currentISO = simResult[0]?.iso || 'ERCOT';
    } else if (simResult.results && typeof simResult.results === 'object') {
        const isos = Object.keys(simResult.results);
        currentISO = isos[0];
    } else {
        currentISO = simResult.iso || 'ERCOT';
    }

    // Render input summary (once, doesn't change per ISO)
    renderInputSummary();

    renderAll();

    // Setup export after first render
    const firstData = getISOResult(currentISO);
    if (firstData) {
        renderRunBar(firstData);
        setupExportButtons(firstData);
    }
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
            <p>Run a simulation from the <a href="/setup">setup page</a> first.</p>
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
    const mode = data.mode || simParams?.mode || 'trajectory';

    // ISO header
    const isoHeaderEl = document.getElementById('isoHeader');
    if (isoHeaderEl) isoHeaderEl.textContent = `${iso} Results`;
    document.getElementById('resultsSubtitle').textContent =
        `${iso} — Market Trajectory Simulation`;

    // Year header above KPI cards
    const kpiYearHeader = document.getElementById('kpiYearHeader');
    if (kpiYearHeader && data.year_results && data.year_results.length > 0) {
        const finalYear = data.year_results[data.year_results.length - 1].year;
        kpiYearHeader.textContent = `${finalYear} Snapshot Results`;
    } else if (kpiYearHeader) {
        kpiYearHeader.textContent = '';
    }

    // Show year drill-down header for trajectory mode
    const yearHeader = document.getElementById('yearDrilldownHeader');
    const hasMultiYear = data.year_results && data.year_results.length > 1;
    if (yearHeader) {
        yearHeader.style.display = hasMultiYear ? '' : 'none';
    }

    // Populate year dropdown with 5-year intervals
    const yearSelect = document.getElementById('yearSelect');
    if (yearSelect && hasMultiYear) {
        const years = data.year_results.map(yr => yr.year);
        // Show 5-year intervals if many years, otherwise all
        const displayYears = years.length > 12
            ? years.filter(y => y % 5 === 0 || y === years[years.length - 1])
            : years;
        yearSelect.innerHTML = displayYears.map(y =>
            `<option value="${y}" ${y === displayYears[displayYears.length - 1] ? 'selected' : ''}>${y}</option>`
        ).join('');
    }

    updateStats(data);
    renderNarrative(data);

    // Level 1: Market-wide
    renderLMPTimeSeries(data);
    renderZonalLMP(data);
    renderSupplyStack(data);
    renderEmissionsByYear(data);
    renderFuelBinTable(data);
    renderMeritOrder(data);

    // Level 2: Year drill-down (removed: profitability, LMP impact, fossil bins)
    renderWhatBuilt(data);
    renderCCSBreakeven(data);
    renderCostLadder(data);
    renderGasShift(data);
    renderNuclearRevenue(data);
    renderSensitivity(data);

    // Level 3: IPP
    renderGeneratorTable(data);

    // Show CCS deep-dive CTA if trajectory mode with year results
    const ccsCta = document.getElementById('ccsCtaCard');
    if (ccsCta) {
        ccsCta.style.display = hasMultiYear ? '' : 'none';
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// STATS
// ══════════════════════════════════════════════════════════════════════════════

function updateStats(data) {
    setText('statCleanPct', fmtPct(data.market_outcome_clean_pct || data.existing_clean_pct));
    setText('statAvgLMP', '$' + fmtNum(data.avg_lmp || data.lmp_summary?.avg));
    setText('statEmissions', fmtNum(data.emissions_mt, 1));
    setText('statDemand', fmtNum(data.demand_twh, 0));
    setText('statNucRev', '$' + fmtNum(data.nuclear_revenue?.total_mwh));

    // Plant-level summary KPI cards
    const plantStats = document.getElementById('plantStats');
    const ps = data.plant_level_summary;
    if (plantStats && ps) {
        setText('statPlantOperating', fmtNum(ps.operating, 0));
        setText('statPlantAtRisk', fmtNum(ps.at_risk, 0));
        setText('statPlantStranded', fmtNum(ps.stranded, 0));
        setText('statPlantTotal', fmtNum(ps.total, 0));
        plantStats.style.display = '';
    } else if (plantStats) {
        plantStats.style.display = 'none';
    }

    // Demand response KPI cards
    const drStats = document.getElementById('drStats');
    if (drStats) {
        const drGwh = data.dr_curtailed_gwh || 0;
        if (drGwh > 0) {
            setText('statDrCurtailed', fmtNum(drGwh, 1));
            setText('statDrPeak', fmtNum(data.dr_peak_gw || 0, 2));
            setText('statDrHours', fmtNum(data.dr_hours || 0, 0));
            setText('statDrPrice', '$' + fmtNum(data.dr_avg_price || 0, 0));
            drStats.style.display = '';
        } else {
            drStats.style.display = 'none';
        }
    }

    // Economic retirement summary
    const retiredMW = data.total_economic_retirement_mw || 0;
    const retiredEl = document.getElementById('statEconRetiredMW');
    if (retiredEl) {
        setText('statEconRetiredMW', fmtNum(retiredMW / 1000, 1) + ' GW');
    }
    const retiredDetail = document.getElementById('econRetirementDetail');
    if (retiredDetail && data.economic_retirements_mw) {
        const parts = Object.entries(data.economic_retirements_mw)
            .filter(([_, mw]) => mw > 0)
            .map(([type, mw]) => `${type.replace('_', ' ')}: ${fmtNum(mw / 1000, 1)} GW`);
        retiredDetail.textContent = parts.join(' · ') || 'None';
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// LEVEL 1: MARKET-WIDE CHARTS
// ══════════════════════════════════════════════════════════════════════════════

// ── Chart 1: LMP + Capacity Revenue Time Series ──
function renderLMPTimeSeries(data) {
    const container = 'chartLMPTimeSeries';
    const traces = [];

    // Check for hourly LMP profile data
    const lmpProfile = data.lmp_time_series || data.lmp_hourly_profile;
    const capRevProfile = data.capacity_rev_time_series || data.capacity_revenue_profile;

    if (lmpProfile && lmpProfile.values && lmpProfile.values.length > 0) {
        // Hourly or representative-day profile (snapshot mode)
        const hours = lmpProfile.hours || lmpProfile.values.map((_, i) => i);
        traces.push({
            x: hours,
            y: lmpProfile.values,
            name: 'Energy LMP ($/MWh)',
            type: 'scatter',
            mode: 'lines',
            line: { color: '#2372B9', width: 2 },
            fill: 'tozeroy',
            fillcolor: 'rgba(35, 114, 185, 0.1)',
        });

        if (capRevProfile && capRevProfile.values && capRevProfile.values.length > 0) {
            traces.push({
                x: capRevProfile.hours || capRevProfile.values.map((_, i) => i),
                y: capRevProfile.values,
                name: 'Capacity Revenue ($/MWh)',
                type: 'scatter',
                mode: 'lines',
                line: { color: '#6BA543', width: 2, dash: 'dash' },
                yaxis: 'y2',
            });
        }

        const xTitle = hours.length > 48 ? 'Hour of Year' : 'Hour of Day';
        Plotly.newPlot(container, traces, {
            ...PLOTLY_LAYOUT_BASE,
            title: { text: `LMP & Capacity Revenue — ${currentISO}`, font: { size: 14 } },
            xaxis: { title: xTitle, gridcolor: '#f0f0f0' },
            yaxis: { title: 'LMP ($/MWh)', gridcolor: '#f0f0f0' },
            yaxis2: capRevProfile ? { title: 'Capacity Rev ($/MWh)', overlaying: 'y', side: 'right' } : undefined,
            legend: { x: 0, y: 1.12, orientation: 'h' },
        }, { responsive: true });
    } else if (data.year_results && data.year_results.length > 1) {
        // Trajectory mode — year-over-year LMP
        const years = data.year_results.map(yr => yr.year);
        const lmps = data.year_results.map(yr => yr.avg_lmp || 0);
        const capRevs = data.year_results.map(yr => yr.capacity_rev_mwh || 0);
        const energyRevs = data.year_results.map(yr => yr.energy_rev_mwh || 0);
        // Use lines-only for annual granularity (many years), markers for sparse years
        const traceMode = years.length > 12 ? 'lines' : 'lines+markers';
        const mSize = years.length > 12 ? 4 : 8;

        traces.push({
            x: years, y: lmps,
            name: 'Avg LMP',
            type: 'scatter', mode: traceMode,
            line: { color: '#2372B9', width: 3 },
            marker: { size: mSize },
        });
        traces.push({
            x: years, y: energyRevs,
            name: 'Energy Revenue',
            type: 'scatter', mode: traceMode,
            line: { color: '#6BA543', width: 2 },
        });
        traces.push({
            x: years, y: capRevs,
            name: 'Capacity Revenue',
            type: 'scatter', mode: traceMode,
            line: { color: '#F47B27', width: 2, dash: 'dash' },
        });

        Plotly.newPlot(container, traces, {
            ...PLOTLY_LAYOUT_BASE,
            title: { text: `LMP & Revenue Trajectory — ${currentISO}`, font: { size: 14 } },
            xaxis: { title: 'Year', gridcolor: '#f0f0f0' },
            yaxis: { title: '$/MWh', gridcolor: '#f0f0f0' },
            legend: { x: 0, y: 1.12, orientation: 'h' },
        }, { responsive: true });
    } else {
        // Fallback: single LMP value
        const avgLMP = data.avg_lmp || 0;
        Plotly.newPlot(container, [{
            x: ['Avg LMP'],
            y: [avgLMP],
            type: 'bar',
            marker: { color: '#2372B9' },
            text: [`$${avgLMP.toFixed(1)}/MWh`],
            textposition: 'outside',
        }], {
            ...PLOTLY_LAYOUT_BASE,
            title: { text: `Average LMP — ${currentISO}`, font: { size: 14 } },
            yaxis: { title: '$/MWh', gridcolor: '#f0f0f0' },
        }, { responsive: true });
    }
}

// ── Chart 1b: Zonal LMP Price Spreads ──
function renderZonalLMP(data) {
    const card = document.getElementById('zonalLMPCard');
    const container = 'chartZonalLMP';
    const tbody = document.getElementById('zonalLMPTableBody');

    const zonalStats = data.zonal_lmp_stats;
    if (!zonalStats || zonalStats.length === 0) {
        if (card) card.style.display = 'none';
        return;
    }

    if (card) card.style.display = '';

    // Sort zones by avg LMP descending
    const sorted = [...zonalStats].sort((a, b) => b.avg_lmp - a.avg_lmp);
    const systemAvg = data.avg_lmp || 0;

    // Plotly horizontal bar chart
    const zoneNames = sorted.map(z => z.zone_name);
    const avgLmps = sorted.map(z => z.avg_lmp);
    const spreads = sorted.map(z => z.price_spread_vs_system);
    const barColors = spreads.map(s =>
        s > 0 ? 'rgba(220, 38, 38, 0.7)' : 'rgba(74, 124, 46, 0.7)');

    const traces = [{
        type: 'bar',
        y: zoneNames,
        x: avgLmps,
        orientation: 'h',
        marker: { color: barColors },
        text: avgLmps.map(v => '$' + v.toFixed(1)),
        textposition: 'outside',
        hovertemplate: '%{y}: $%{x:.1f}/MWh<extra></extra>',
    }];

    // System average reference line
    const shapes = [{
        type: 'line',
        x0: systemAvg, x1: systemAvg,
        y0: -0.5, y1: zoneNames.length - 0.5,
        line: { color: '#1A232F', width: 2, dash: 'dash' },
    }];

    const annotations = [{
        x: systemAvg, y: zoneNames.length - 0.3,
        text: `System Avg $${systemAvg.toFixed(1)}`,
        showarrow: false, font: { size: 11, color: '#1A232F' },
        xanchor: 'left', xshift: 5,
    }];

    Plotly.newPlot(container, traces, {
        ...PLOTLY_LAYOUT_BASE,
        title: { text: 'Average LMP by Transmission Zone', font: { size: 14 } },
        xaxis: { title: '$/MWh', zeroline: true },
        yaxis: { automargin: true },
        shapes, annotations,
        margin: { l: 120, r: 60, t: 40, b: 40 },
        height: Math.max(200, zoneNames.length * 50 + 80),
    }, { responsive: true });

    // Populate table
    if (tbody) {
        tbody.innerHTML = sorted.map(z => {
            const spreadClass = z.price_spread_vs_system > 0 ? 'color:#DC2626' : 'color:#4A7C2E';
            const spreadSign = z.price_spread_vs_system > 0 ? '+' : '';
            return `<tr>
                <td><strong>${z.zone_name}</strong></td>
                <td>$${z.avg_lmp.toFixed(1)}</td>
                <td>$${z.peak_lmp.toFixed(1)}</td>
                <td>$${z.offpeak_lmp.toFixed(1)}</td>
                <td>$${z.p10_lmp.toFixed(1)}</td>
                <td>$${z.p90_lmp.toFixed(1)}</td>
                <td style="${spreadClass}; font-weight:600;">${spreadSign}$${z.price_spread_vs_system.toFixed(1)}</td>
            </tr>`;
        }).join('');
    }
}

// ── Chart 2: Supply Stack (stacked bars at 5-year intervals) ──
function renderSupplyStack(data) {
    const container = 'chartSupplyStack';
    const tableContainer = document.getElementById('supplyStackTableContainer');

    if (data.year_results && data.year_results.length > 1) {
        // Trajectory: stacked bars at 5-year intervals
        const allYears = data.year_results.map(yr => yr.year);
        // Filter to 5-year intervals
        const displayYears = allYears.length > 12
            ? allYears.filter(y => y % 5 === 0 || y === allYears[allYears.length - 1])
            : allYears;
        const displayResults = data.year_results.filter(yr => displayYears.includes(yr.year));

        const allResources = new Set();
        displayResults.forEach(yr => {
            Object.keys(yr.resource_mix_twh || {}).forEach(r => allResources.add(r));
        });

        const traces = [];
        for (const res of allResources) {
            traces.push({
                x: displayYears.map(String),
                y: displayResults.map(yr => (yr.resource_mix_twh || {})[res] || 0),
                name: res.replace(/_/g, ' '),
                type: 'bar',
                marker: { color: ALL_RESOURCE_COLORS[res] || '#9CA3AF' },
                hovertemplate: `${res.replace(/_/g, ' ')}: %{y:.1f} TWh<extra></extra>`,
            });
        }

        Plotly.newPlot(container, traces, {
            ...PLOTLY_LAYOUT_BASE,
            title: { text: `Supply Stack by Year — ${currentISO}`, font: { size: 14 } },
            barmode: 'stack',
            xaxis: { title: 'Year', type: 'category' },
            yaxis: { title: 'Generation (TWh)', gridcolor: '#f0f0f0' },
            legend: { x: 0, y: 1.15, orientation: 'h' },
        }, { responsive: true });

        // Build data table
        if (tableContainer) {
            const resList = [...allResources];
            const thead = document.getElementById('supplyStackTableHead');
            const tbody = document.getElementById('supplyStackTableBody');
            thead.innerHTML = `<tr><th>Resource</th>${displayYears.map(y => `<th>${y}</th>`).join('')}</tr>`;
            tbody.innerHTML = resList.map(res =>
                `<tr><td style="font-weight:600;">${res.replace(/_/g, ' ')}</td>${displayResults.map(yr =>
                    `<td>${fmtNum((yr.resource_mix_twh || {})[res] || 0, 1)}</td>`
                ).join('')}</tr>`
            ).join('');
            tableContainer.style.display = '';
        }

        // Show download button for annual data if many years
        if (allYears.length > displayYears.length && tableContainer) {
            const existing = tableContainer.querySelector('.annual-note');
            if (!existing) {
                const note = document.createElement('div');
                note.className = 'annual-note';
                note.style.cssText = 'margin-top: 0.5rem; font-size: 0.85rem; color: #6B7280;';
                note.textContent = `Showing ${displayYears.length} of ${allYears.length} years (5-year intervals). Full annual data available in CSV export.`;
                tableContainer.appendChild(note);
            }
        }
    } else if (data.resource_mix_twh && Object.keys(data.resource_mix_twh).length > 0) {
        const resources = Object.keys(data.resource_mix_twh);
        const values = resources.map(r => data.resource_mix_twh[r]);
        const colors = resources.map(r => ALL_RESOURCE_COLORS[r] || '#9CA3AF');

        Plotly.newPlot(container, [{
            x: resources.map(r => r.replace(/_/g, ' ')),
            y: values,
            type: 'bar',
            marker: { color: colors },
            hovertemplate: '%{x}: %{y:.1f} TWh<extra></extra>',
        }], {
            ...PLOTLY_LAYOUT_BASE,
            title: { text: `Resource Mix — ${currentISO}`, font: { size: 14 } },
            yaxis: { title: 'Generation (TWh)', gridcolor: '#f0f0f0' },
        }, { responsive: true });
        if (tableContainer) tableContainer.style.display = 'none';
    } else {
        placeholderChart(container);
        if (tableContainer) tableContainer.style.display = 'none';
    }
}

// ── Chart 2b: System Emissions by Year ──
const FUEL_EMISSION_COLORS = {
    coal_steam: '#374151',
    gas_ccgt: '#6B7280',
    gas_ct: '#9CA3AF',
    oil_ct: '#92400E',
};
const FUEL_EMISSION_LABELS = {
    coal_steam: 'Coal',
    gas_ccgt: 'Gas CCGT',
    gas_ct: 'Gas CT',
    oil_ct: 'Oil',
};

function renderEmissionsByYear(data) {
    const card = document.getElementById('emissionsByYearCard');
    const container = document.getElementById('chartEmissionsByYear');
    const statsEl = document.getElementById('emissionsHeadlineStats');
    if (!card || !container) return;

    const eData = data.emissions_by_fuel_by_year;
    if (!eData || !eData.years || eData.years.length < 2) {
        card.style.display = 'none';
        return;
    }

    card.style.display = '';
    const years = eData.years;
    const fuelTypes = Object.keys(eData).filter(k => k !== 'years');
    const traceMode = years.length > 12 ? 'lines' : 'lines+markers';

    // Stacked area traces
    const traces = [];
    for (const fuel of fuelTypes) {
        traces.push({
            x: years,
            y: eData[fuel],
            name: FUEL_EMISSION_LABELS[fuel] || fuel.replace(/_/g, ' '),
            type: 'scatter',
            mode: 'lines',
            stackgroup: 'emissions',
            fillcolor: FUEL_EMISSION_COLORS[fuel] || '#9CA3AF',
            line: { width: 0.5, color: FUEL_EMISSION_COLORS[fuel] || '#9CA3AF' },
            hovertemplate: '%{fullData.name}: %{y:.1f} Mt<extra></extra>',
        });
    }

    // Total emissions line overlay
    const totals = years.map((_, i) =>
        fuelTypes.reduce((sum, fuel) => sum + (eData[fuel][i] || 0), 0)
    );
    traces.push({
        x: years,
        y: totals,
        name: 'Total',
        type: 'scatter',
        mode: traceMode,
        line: { color: '#EF4444', width: 3 },
        marker: { size: years.length > 12 ? 4 : 8 },
        hovertemplate: 'Total: %{y:.1f} Mt<extra></extra>',
    });

    Plotly.newPlot(container, traces, {
        ...PLOTLY_LAYOUT_BASE,
        title: { text: `System CO₂ Emissions by Fuel Type — ${currentISO}`, font: { size: 14 } },
        xaxis: { title: 'Year', gridcolor: '#f0f0f0' },
        yaxis: { title: 'CO₂ Emissions (Mt)', gridcolor: '#f0f0f0' },
        legend: { x: 0, y: 1.15, orientation: 'h' },
    }, { responsive: true });

    // Headline stats
    if (statsEl && totals.length >= 2) {
        const first = totals[0];
        const last = totals[totals.length - 1];
        const pctChange = first > 0 ? ((last - first) / first * 100).toFixed(0) : 0;
        statsEl.innerHTML = `
            <div style="text-align: center; min-width: 120px;">
                <div style="font-size: 1.5rem; font-weight: 700; color: var(--navy);">${first.toFixed(1)} Mt</div>
                <div style="font-size: 0.85rem; color: var(--text-muted);">${years[0]} Emissions</div>
            </div>
            <div style="text-align: center; min-width: 120px;">
                <div style="font-size: 1.5rem; font-weight: 700; color: ${last < first ? '#22C55E' : '#EF4444'};">${last.toFixed(1)} Mt</div>
                <div style="font-size: 0.85rem; color: var(--text-muted);">${years[years.length-1]} Emissions</div>
            </div>
            <div style="text-align: center; min-width: 120px;">
                <div style="font-size: 1.5rem; font-weight: 700; color: ${pctChange < 0 ? '#22C55E' : '#EF4444'};">${pctChange}%</div>
                <div style="font-size: 0.85rem; color: var(--text-muted);">Change</div>
            </div>
        `;
    }
}

// ── Table 3: Fuel Source / Heat Rate Bin Breakdown ──
function renderFuelBinTable(data) {
    const tbody = document.getElementById('fuelBinTableBody');
    if (!tbody) return;

    const bins = data.fuel_bin_table || [];

    if (bins.length > 0) {
        tbody.innerHTML = bins.map(row => `
            <tr>
                <td style="font-weight: 600;">${row.fuel_type.replace(/_/g, ' ')}</td>
                <td>${row.heat_rate_bin || '—'}</td>
                <td>${fmtNum(row.capacity_gw, 1)}</td>
                <td>${fmtPct((row.capacity_factor || 0) * 100)}</td>
                <td>${fmtNum(row.generation_twh, 1)}</td>
                <td>$${fmtNum(row.marginal_cost, 1)}</td>
                <td>$${fmtNum(row.avg_revenue, 1)}</td>
                <td class="status-cell ${row.status || 'operating'}">${row.status || 'operating'}</td>
            </tr>
        `).join('');
    } else {
        // Build from generator_economics data
        const gens = data.generator_economics || [];
        if (gens.length > 0) {
            tbody.innerHTML = gens.map(g => {
                const capGW = (g.capacity_mw || 0) / 1000;
                const cf = g.capacity_factor || 0;
                const genTWh = capGW * cf * 8.76; // GW × CF × 8760h / 1000
                return `
                    <tr>
                        <td style="font-weight: 600;">${g.unit_type.replace(/_/g, ' ')}</td>
                        <td>—</td>
                        <td>${fmtNum(capGW, 1)}</td>
                        <td>${fmtPct(cf * 100)}</td>
                        <td>${fmtNum(genTWh, 1)}</td>
                        <td>$${fmtNum(g.marginal_cost, 1)}</td>
                        <td>$${fmtNum(g.avg_revenue_mwh, 1)}</td>
                        <td class="status-cell ${g.status || 'operating'}">${g.status || 'operating'}</td>
                    </tr>
                `;
            }).join('');
        } else {
            tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; color:#9CA3AF;">No fuel bin data available</td></tr>';
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// EXISTING CHARTS (Level 2 + 3) — retained from previous version
// ══════════════════════════════════════════════════════════════════════════════

// ── Merit-Order Stack (time series with heat rate bins) ──
const HEAT_RATE_BIN_COLORS = {
    'coal_steam': '#374151',
    'gas_ccgt < 7.2': '#1B5E9E',
    'gas_ccgt 7.2-8.0': '#2372B9',
    'gas_ccgt > 8.0': '#5BA3E0',
    'gas_ct': '#007FA4',
    'oil_ct': '#92400E',
};

function renderMeritOrder(data) {
    const container = 'chartMeritOrder';
    const tableContainer = document.getElementById('meritOrderTableContainer');

    if (data.year_results && data.year_results.length > 1) {
        // Trajectory: stacked bars at 5-year intervals showing capacity by fuel/heat rate bin
        const allYears = data.year_results.map(yr => yr.year);
        const displayYears = allYears.length > 12
            ? allYears.filter(y => y % 5 === 0 || y === allYears[allYears.length - 1])
            : allYears;
        const displayResults = data.year_results.filter(yr => displayYears.includes(yr.year));

        // Collect all fuel bins across years
        const allBins = new Set();
        displayResults.forEach(yr => {
            const bins = yr.fuel_bin_table || yr.generator_economics || [];
            bins.forEach(b => {
                const key = b.heat_rate_bin && b.heat_rate_bin !== '—'
                    ? `${b.fuel_type || b.unit_type} ${b.heat_rate_bin}`
                    : (b.fuel_type || b.unit_type);
                allBins.add(key);
            });
        });

        const binList = [...allBins];
        const defaultColors = ['#374151', '#1B5E9E', '#2372B9', '#5BA3E0', '#007FA4', '#92400E', '#6B7280', '#9CA3AF'];
        const traces = binList.map((bin, idx) => ({
            x: displayYears.map(String),
            y: displayResults.map(yr => {
                const bins = yr.fuel_bin_table || yr.generator_economics || [];
                return bins.reduce((sum, b) => {
                    const key = b.heat_rate_bin && b.heat_rate_bin !== '—'
                        ? `${b.fuel_type || b.unit_type} ${b.heat_rate_bin}`
                        : (b.fuel_type || b.unit_type);
                    return key === bin ? sum + (b.capacity_gw || (b.capacity_mw || 0) / 1000) : sum;
                }, 0);
            }),
            name: bin.replace(/_/g, ' '),
            type: 'bar',
            marker: { color: HEAT_RATE_BIN_COLORS[bin] || defaultColors[idx % defaultColors.length] },
        }));

        Plotly.newPlot(container, traces, {
            ...PLOTLY_LAYOUT_BASE,
            title: { text: `Merit-Order by Heat Rate Bin — ${currentISO}`, font: { size: 14 } },
            barmode: 'stack',
            xaxis: { title: 'Year', type: 'category' },
            yaxis: { title: 'Capacity (GW)', gridcolor: '#f0f0f0' },
            legend: { x: 0, y: 1.15, orientation: 'h' },
        }, { responsive: true });

        // Build table
        if (tableContainer) {
            const thead = document.getElementById('meritOrderTableHead');
            const tbody = document.getElementById('meritOrderTableBody');
            thead.innerHTML = `<tr><th>Fuel / Heat Rate Bin</th>${displayYears.map(y => `<th>${y} (GW)</th>`).join('')}</tr>`;
            tbody.innerHTML = binList.map(bin => {
                return `<tr><td style="font-weight:600;">${bin.replace(/_/g, ' ')}</td>${displayResults.map(yr => {
                    const bins = yr.fuel_bin_table || yr.generator_economics || [];
                    const val = bins.reduce((sum, b) => {
                        const key = b.heat_rate_bin && b.heat_rate_bin !== '—'
                            ? `${b.fuel_type || b.unit_type} ${b.heat_rate_bin}`
                            : (b.fuel_type || b.unit_type);
                        return key === bin ? sum + (b.capacity_gw || (b.capacity_mw || 0) / 1000) : sum;
                    }, 0);
                    return `<td>${fmtNum(val, 1)}</td>`;
                }).join('')}</tr>`;
            }).join('');
            tableContainer.style.display = '';
        }
    } else {
        // Single-year: standard merit order bar
        const gens = data.generator_economics || data.merit_order_stack || [];
        if (!gens.length) { placeholderChart(container); if (tableContainer) tableContainer.style.display = 'none'; return; }

        const sorted = [...gens].sort((a, b) => a.marginal_cost - b.marginal_cost);
        Plotly.newPlot(container, [{
            x: sorted.map(g => (g.unit_type || '').replace(/_/g, ' ')),
            y: sorted.map(g => (g.capacity_mw || 0) / 1000),
            marker: { color: sorted.map(g => UNIT_COLORS[g.unit_type] || '#9CA3AF') },
            type: 'bar',
            name: 'Capacity (GW)',
            hovertemplate: '%{x}<br>%{y:.1f} GW<br>MC: $%{customdata:.1f}/MWh<extra></extra>',
            customdata: sorted.map(g => g.marginal_cost),
        }], {
            ...PLOTLY_LAYOUT_BASE,
            title: { text: `Merit-Order Stack — ${currentISO}`, font: { size: 14 } },
            yaxis: { title: 'Capacity (GW)', gridcolor: '#f0f0f0' },
        }, { responsive: true });
        if (tableContainer) tableContainer.style.display = 'none';
    }
}

// ── Nuclear Revenue ──
function renderNuclearRevenue(data) {
    const nuc = data.nuclear_revenue;
    if (!nuc) { placeholderChart('chartNuclearRevenue'); return; }

    const categories = ['Energy', 'Capacity', 'PTC', 'Total'];
    const values = [nuc.energy_rev_mwh, nuc.capacity_rev_mwh, nuc.ptc_mwh, nuc.total_mwh];
    const colors = ['#2372B9', '#007FA4', '#6BA543', '#1A232F'];

    Plotly.newPlot('chartNuclearRevenue', [{
        x: categories, y: values, type: 'bar',
        marker: { color: colors },
        hovertemplate: '%{x}: $%{y:.1f}/MWh<extra></extra>',
    }], {
        ...PLOTLY_LAYOUT_BASE,
        title: { text: `Nuclear Revenue Stack — ${currentISO}`, font: { size: 14 } },
        yaxis: { title: '$/MWh', gridcolor: '#f0f0f0' },
    }, { responsive: true });
}

// ── CCS Breakeven ──
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
        { x: carbonPrices, y: newGasCost, name: 'New Efficient Gas', mode: 'lines', line: { color: '#6BA543', width: 3 } },
    ], {
        ...PLOTLY_LAYOUT_BASE,
        title: { text: `CCS Retrofit vs New Gas — ${currentISO}`, font: { size: 14 } },
        xaxis: { title: 'Carbon Price ($/ton)', gridcolor: '#f0f0f0' },
        yaxis: { title: '$/MWh Total Cost', gridcolor: '#f0f0f0' },
    }, { responsive: true });
}

// ── What Gets Built (donut chart) ──
function renderWhatBuilt(data) {
    const built = data.what_gets_built || data.deployment_stack;
    if (!built) { placeholderChart('chartWhatBuilt'); return; }

    const resources = Object.keys(built).filter(r => built[r] > 0);
    const values = resources.map(r => built[r]);
    const colors = resources.map(r => CLEAN_COLORS[r] || '#9CA3AF');

    Plotly.newPlot('chartWhatBuilt', [{
        labels: resources.map(r => r.replace(/_/g, ' ')),
        values: values,
        type: 'pie',
        hole: 0.45,
        marker: { colors: colors },
        textinfo: 'label+percent',
        hovertemplate: '%{label}: %{value:.1f} GW (%{percent})<extra></extra>',
    }], {
        ...PLOTLY_LAYOUT_BASE,
        title: { text: `What Gets Built — ${currentISO}`, font: { size: 14 } },
    }, { responsive: true });
}

// ── Cost Ladder ──
function renderCostLadder(data) {
    const ladder = data.cost_ladder;
    if (!ladder) { placeholderChart('chartCostLadder'); return; }

    const cumGW = ladder.map(s => s.cumulative_gw);
    const costs = ladder.map(s => s.cost_mwh);
    const revenues = ladder.map(s => s.revenue_mwh);

    Plotly.newPlot('chartCostLadder', [
        { x: cumGW, y: costs, name: 'Cost $/MWh', type: 'scatter', mode: 'lines+markers', line: { color: '#F47B27', width: 3 } },
        { x: cumGW, y: revenues, name: 'Revenue $/MWh', type: 'scatter', mode: 'lines+markers', line: { color: '#6BA543', width: 3 } },
    ], {
        ...PLOTLY_LAYOUT_BASE,
        title: { text: `Clean Energy Cost Ladder — ${currentISO}`, font: { size: 14 } },
        xaxis: { title: 'Cumulative New Clean GW', gridcolor: '#f0f0f0' },
        yaxis: { title: '$/MWh', gridcolor: '#f0f0f0' },
    }, { responsive: true });
}

// ── Gas Shift ──
function renderGasShift(data) {
    const shift = data.gas_fleet_shift || data.fleet_shift;
    if (!shift) { placeholderChart('chartGasShift'); return; }

    if (Array.isArray(shift)) {
        Plotly.newPlot('chartGasShift', [
            { x: shift.map(s => s.carbon_price), y: shift.map(s => s.efficient_cf * 100), name: 'Efficient CCGT', mode: 'lines', line: { color: '#6BA543', width: 3 } },
            { x: shift.map(s => s.carbon_price), y: shift.map(s => s.avg_cf * 100), name: 'Average CCGT', mode: 'lines', line: { color: '#6B7280', width: 3 } },
            { x: shift.map(s => s.carbon_price), y: shift.map(s => s.old_cf * 100), name: 'Old CCGT', mode: 'lines', line: { color: '#9CA3AF', width: 3, dash: 'dash' } },
        ], {
            ...PLOTLY_LAYOUT_BASE,
            title: { text: `Gas Fleet Efficiency Shift — ${currentISO}`, font: { size: 14 } },
            xaxis: { title: 'Carbon Price ($/ton)', gridcolor: '#f0f0f0' },
            yaxis: { title: 'Capacity Factor %', gridcolor: '#f0f0f0', range: [0, 100] },
        }, { responsive: true });
    } else {
        placeholderChart('chartGasShift');
    }
}

// ── Sensitivity Heatmap ──
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
        title: { text: `Gas vs Carbon Price Sensitivity — ${currentISO}`, font: { size: 14 } },
        xaxis: { title: 'Gas Price ($/MMBtu)' },
        yaxis: { title: 'Carbon Price ($/ton)' },
    }, { responsive: true });
}

// ── Generator Table (Level 3: IPP) ──
function renderGeneratorTable(data) {
    const gens = data.generator_economics || [];
    const tbody = document.getElementById('generatorTableBody');
    if (!tbody) return;

    tbody.innerHTML = gens.map(g => `
        <tr>
            <td>${g.unit_type.replace(/_/g, ' ')}</td>
            <td>${fmtNum((g.capacity_mw || 0) / 1000, 1)}</td>
            <td>$${fmtNum(g.marginal_cost)}</td>
            <td>${fmtNum(g.dispatch_hours, 0)}</td>
            <td>${fmtPct(g.capacity_factor * 100)}</td>
            <td>$${fmtNum(g.avg_revenue_mwh)}</td>
            <td style="color: ${g.profit_mwh >= 0 ? '#4A7C2E' : '#991b1b'}">$${fmtNum(g.profit_mwh)}</td>
            <td><span class="status-badge status-${g.status || 'marginal'}">${g.status || 'unknown'}</span></td>
        </tr>
    `).join('');
}

// ══════════════════════════════════════════════════════════════════════════════
// HELPERS
// ══════════════════════════════════════════════════════════════════════════════

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
        el.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#7F8F97;font-style:italic;">No data for this chart</div>';
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// INPUT SUMMARY, NARRATIVE, RUN BAR, EXPORT
// ══════════════════════════════════════════════════════════════════════════════

function renderInputSummary() {
    if (!simParams || Object.keys(simParams).length === 0) return;

    const grid = document.getElementById('inputSummaryGrid');
    if (!grid) return;

    const items = [];

    // Mode & ISO
    items.push({ label: 'Mode', value: simParams.mode || 'trajectory' });
    const isos = simParams.isos || [simParams.iso || 'ERCOT'];
    items.push({ label: 'ISO(s)', value: isos.join(', ') });

    // Fuel prices
    const fp = simParams.fuel_prices || {};
    items.push({ label: 'Gas Price', value: `$${fp.gas || 3.5}/MMBtu` });
    items.push({ label: 'Coal Price', value: `$${fp.coal || 2.25}/MMBtu` });
    items.push({ label: 'Oil Price', value: `$${fp.oil || 10.5}/MMBtu` });

    // Carbon
    items.push({ label: 'Carbon Price', value: `$${simParams.carbon_price || 5.5}/ton` });

    // Clean LCOEs
    const lcoe = simParams.clean_lcoes || {};
    items.push({ label: 'Solar LCOE', value: `$${lcoe.solar || '--'}/MWh` });
    items.push({ label: 'Wind LCOE', value: `$${lcoe.wind || '--'}/MWh` });
    items.push({ label: 'Nuclear LCOE', value: `$${lcoe.nuclear || '--'}/MWh` });
    items.push({ label: 'CCS LCOE', value: `$${lcoe.ccs_ccgt || '--'}/MWh` });

    // Key toggles
    items.push({ label: 'Demand Growth', value: simParams.demand_growth || 'Medium' });
    items.push({ label: 'Transmission', value: simParams.transmission_level || 'Medium' });
    items.push({ label: '45Q', value: simParams.q45 ? 'On' : 'Off' });

    // Storage
    const sc = simParams.storage_costs || {};
    items.push({ label: 'Battery 4hr', value: `$${sc.battery || 295}/kW-yr` });
    items.push({ label: 'LDES 100hr', value: `$${sc.ldes || 220}/kW-yr` });

    // Nuclear
    items.push({ label: 'Nuc Retirement', value: `$${simParams.nuclear_retirement_threshold || 30}/MWh` });

    grid.innerHTML = items.map(i =>
        `<div class="input-summary-item"><span class="label">${i.label}</span><span class="value">${i.value}</span></div>`
    ).join('');
}

function renderNarrative(data) {
    const box = document.getElementById('narrativeBox');
    const text = document.getElementById('narrativeText');
    if (!box || !text) return;

    const narrative = data.narrative;
    if (narrative) {
        text.textContent = narrative;
        box.style.display = '';
    } else {
        box.style.display = 'none';
    }
}

function renderRunBar(data) {
    const bar = document.getElementById('runBar');
    const badge = document.getElementById('runIdBadge');
    if (!bar || !badge) return;

    if (data.run_id) {
        badge.textContent = `Run: ${data.run_id}`;
        bar.style.display = '';
    } else {
        bar.style.display = 'none';
    }
}

function setupExportButtons(data) {
    // Export Charts button
    const btnExport = document.getElementById('btnExportCharts');
    if (btnExport) {
        btnExport.addEventListener('click', async () => {
            if (!data.run_id) {
                alert('No run ID available for export.');
                return;
            }
            btnExport.textContent = 'Exporting...';
            btnExport.disabled = true;

            const chartContainers = document.querySelectorAll('.chart-container, .chart-container-lg');
            let saved = 0;

            for (const container of chartContainers) {
                const chartId = container.id || `chart_${saved}`;
                try {
                    if (typeof html2canvas === 'function') {
                        const canvas = await html2canvas(container, { backgroundColor: '#ffffff', scale: 2 });
                        const imageData = canvas.toDataURL('image/png');
                        await fetch(`/api/runs/${data.run_id}/save-chart`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ name: chartId, image: imageData }),
                        });
                        saved++;
                    }
                } catch (e) {
                    console.warn(`Failed to export chart ${chartId}:`, e);
                }
            }

            btnExport.textContent = `Exported ${saved} charts`;
            setTimeout(() => {
                btnExport.textContent = 'Export Charts';
                btnExport.disabled = false;
            }, 2000);
        });
    }

    // Download Plant-Level CSV button
    const btnPlantCSV = document.getElementById('btnDownloadPlantCSV');
    if (btnPlantCSV && data.run_id) {
        btnPlantCSV.addEventListener('click', async () => {
            try {
                const resp = await fetch(`/api/runs/${data.run_id}/plant-csv`);
                if (!resp.ok) {
                    const err = await resp.json().catch(() => ({}));
                    alert(err.detail || 'Plant-level CSV not available for this run.');
                    return;
                }
                const blob = await resp.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${data.run_id}_plant_results.csv`;
                a.click();
                URL.revokeObjectURL(url);
            } catch (e) {
                console.warn('Failed to download plant CSV:', e);
                alert('Failed to download plant-level CSV.');
            }
        });
    } else if (btnPlantCSV) {
        btnPlantCSV.disabled = true;
        btnPlantCSV.title = 'No run ID — run a simulation first';
    }

    // Download CSV button
    const btnCSV = document.getElementById('btnDownloadCSV');
    if (btnCSV && data.run_id) {
        btnCSV.addEventListener('click', async () => {
            try {
                const resp = await fetch(`/api/runs/${data.run_id}`);
                const runData = await resp.json();
                // Trigger download of the inputs text as a file
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${data.run_id}_results.json`;
                a.click();
                URL.revokeObjectURL(url);
            } catch (e) {
                console.warn('Failed to download:', e);
            }
        });
    }
}

// ── Initialize ──
// Guard: wait for Plotly to be available (handles async CDN fallback edge case)
function waitForPlotlyAndInit() {
    if (typeof Plotly !== 'undefined') {
        init();
    } else {
        setTimeout(waitForPlotlyAndInit, 100);
    }
}
waitForPlotlyAndInit();
