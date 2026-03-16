/**
 * Gas Lock-In & Buyer Influence — Cross-Company Analysis
 * =======================================================
 * Aggregates gas development data from all 7 IPPs and renders
 * comparative charts showing gas lock-in risk and the impact
 * of PPA market depth on fleet trajectories.
 *
 * Data source: IPP_SMARTARGETS_DATA (ipp-smartargets-data.js)
 */

/* global IPP_SMARTARGETS_DATA, Chart, RESOURCE_COLORS, ISO_COLORS */

(function () {
    'use strict';

    if (typeof IPP_SMARTARGETS_DATA === 'undefined') return;

    const data = IPP_SMARTARGETS_DATA;
    const companyIds = Object.keys(data.companies);

    const COMPANY_COLORS = [
        '#6366F1', '#22C55E', '#F59E0B', '#E91E63', '#0EA5E9', '#9C27B0', '#14B8A6'
    ];

    // Gas development drivers by ISO (same as ipp-company-report.js)
    const ISO_DRIVERS = {
        ERCOT: { pct: 0.15 }, PJM: { pct: 0.10 }, CAISO: { pct: 0.05 },
        NYISO: { pct: 0 }, NEISO: { pct: 0.08 }, MISO: { pct: 0.10 }, SPP: { pct: 0 }
    };

    function fmt(n, d) {
        if (n === undefined || n === null) return '—';
        if (d === undefined) d = Math.abs(n) >= 100 ? 0 : 1;
        return n.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    }

    function toRGBA(color, alpha) {
        if (!color || typeof color !== 'string') return color;
        if (color.startsWith('rgba(')) return color.replace(/,\s*[\d.]+\)$/, ', ' + alpha + ')');
        if (color.startsWith('#')) {
            var hex = color.slice(1);
            if (hex.length === 3) hex = hex[0]+hex[0]+hex[1]+hex[1]+hex[2]+hex[2];
            return 'rgba(' + parseInt(hex.slice(0,2),16) + ',' + parseInt(hex.slice(2,4),16) + ',' + parseInt(hex.slice(4,6),16) + ',' + alpha + ')';
        }
        return color;
    }

    // Compute gas development data per company
    var companyGasData = companyIds.map(function (id, idx) {
        var co = data.companies[id];
        var fleet = co.fleet_summary || {};
        var gasCap = 0;
        var gasISOs = {};

        for (var iso in fleet) {
            var isoGas = (fleet[iso].gas_ccgt ? fleet[iso].gas_ccgt.cap_mw : 0) +
                         (fleet[iso].gas_peaker ? fleet[iso].gas_peaker.cap_mw : 0);
            if (isoGas > 0) {
                gasISOs[iso] = isoGas;
                gasCap += isoGas;
            }
        }

        // New gas per ISO
        var newGasMW = 0;
        var newGasByISO = {};
        for (var iso2 in gasISOs) {
            var pct = ISO_DRIVERS[iso2] ? ISO_DRIVERS[iso2].pct : 0;
            var newMW = Math.round(gasISOs[iso2] * pct);
            newGasByISO[iso2] = newMW;
            newGasMW += newMW;
        }

        // Annual CO2 from new gas: MW * 0.4 tCO2/MWh * 50% CF * 8760h / 1e6
        var annualCO2 = newGasMW * 0.4 * 0.5 * 8.76 / 1000;
        // Cumulative over 25 years with declining dispatch
        var cumulativeCO2 = annualCO2 * (1.0 + 0.9 + 0.7 + 0.5 + 0.3) * 5; // 5-year blocks

        // PPA level dimension data for counterfactual
        var dims = co.dimension_fans;
        var lowPPA_2040 = dims && dims.ppa_level && dims.ppa_level.Low ?
            dims.ppa_level.Low.emissions.p50[3] : null;
        var highPPA_2040 = dims && dims.ppa_level && dims.ppa_level.High ?
            dims.ppa_level.High.emissions.p50[3] : null;

        return {
            id: id,
            name: co.shortName || co.name,
            color: COMPANY_COLORS[idx % COMPANY_COLORS.length],
            gasCap: gasCap,
            newGasMW: newGasMW,
            newGasByISO: newGasByISO,
            annualCO2: annualCO2,
            cumulativeCO2: cumulativeCO2,
            lowPPA_2040: lowPPA_2040,
            highPPA_2040: highPPA_2040,
            ppaDelta: (lowPPA_2040 !== null && highPPA_2040 !== null) ? lowPPA_2040 - highPPA_2040 : 0
        };
    });

    // Sort by new gas MW descending for exposure ranking
    var ranked = companyGasData.slice().sort(function (a, b) { return b.newGasMW - a.newGasMW; });

    // ─── Section 2: Gas Exposure by Company ──────────────────

    var exposureCtx = document.getElementById('gasExposureChart');
    if (exposureCtx) {
        new Chart(exposureCtx, {
            type: 'bar',
            data: {
                labels: ranked.map(function (c) { return c.name; }),
                datasets: [{
                    label: 'Existing Gas (MW)',
                    data: ranked.map(function (c) { return c.gasCap; }),
                    backgroundColor: ranked.map(function (c) { return toRGBA(c.color, 0.15); }),
                    borderColor: ranked.map(function (c) { return c.color; }),
                    borderWidth: 1.5,
                    borderRadius: 4
                }, {
                    label: 'Potential New Gas (MW)',
                    data: ranked.map(function (c) { return c.newGasMW; }),
                    backgroundColor: ranked.map(function () { return toRGBA('#EF4444', 0.25); }),
                    borderColor: '#EF4444',
                    borderWidth: 1.5,
                    borderRadius: 4
                }]
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
                    legend: { position: 'bottom', labels: { usePointStyle: true, padding: 12 } }
                }
            }
        });
    }

    // Gas by ISO chart
    var isoCtx = document.getElementById('gasByISOChart');
    if (isoCtx) {
        var allISOs = ['ERCOT', 'PJM', 'MISO', 'NEISO', 'CAISO'];
        var isoTotals = allISOs.map(function (iso) {
            return companyGasData.reduce(function (s, c) { return s + (c.newGasByISO[iso] || 0); }, 0);
        });

        new Chart(isoCtx, {
            type: 'bar',
            data: {
                labels: allISOs,
                datasets: [{
                    label: 'Total New Gas (MW)',
                    data: isoTotals,
                    backgroundColor: allISOs.map(function (iso) { return toRGBA(ISO_COLORS[iso] || '#6B7280', 0.25); }),
                    borderColor: allISOs.map(function (iso) { return ISO_COLORS[iso] || '#6B7280'; }),
                    borderWidth: 1.5,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { title: { display: true, text: 'New Gas (MW)' } },
                    x: { grid: { display: false } }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    // ─── Section 3: Lock-in cost stats + cumulative emissions ─────

    var statsEl = document.getElementById('lockinStats');
    if (statsEl) {
        var totalNewGW = companyGasData.reduce(function (s, c) { return s + c.newGasMW; }, 0) / 1000;
        var totalCumCO2 = companyGasData.reduce(function (s, c) { return s + c.cumulativeCO2; }, 0);
        var totalAnnualCO2 = companyGasData.reduce(function (s, c) { return s + c.annualCO2; }, 0);
        // Rough stranded asset value: $1.2M/MW for new gas
        var strandedValue = totalNewGW * 1200;

        statsEl.innerHTML = '<div class="stat-card"><div class="stat-value">' + fmt(totalNewGW, 1) + ' GW</div><div class="stat-label">Total New Gas at Risk</div></div>' +
            '<div class="stat-card"><div class="stat-value">' + fmt(totalAnnualCO2, 1) + ' Mt/yr</div><div class="stat-label">Annual CO₂ from New Gas</div></div>' +
            '<div class="stat-card"><div class="stat-value">' + fmt(totalCumCO2, 0) + ' Mt</div><div class="stat-label">Cumulative CO₂ through 2050</div></div>' +
            '<div class="stat-card"><div class="stat-value">$' + fmt(strandedValue, 0) + 'M</div><div class="stat-label">Stranded Asset Value</div></div>';
    }

    var cumCtx = document.getElementById('cumulativeEmissionsChart');
    if (cumCtx) {
        new Chart(cumCtx, {
            type: 'bar',
            data: {
                labels: ranked.map(function (c) { return c.name; }),
                datasets: [{
                    label: 'Cumulative Locked-In CO₂ (Mt)',
                    data: ranked.map(function (c) { return +c.cumulativeCO2.toFixed(1); }),
                    backgroundColor: ranked.map(function (c) { return toRGBA('#EF4444', 0.25); }),
                    borderColor: '#EF4444',
                    borderWidth: 1.5,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: 'Mt CO₂ (2025-2050)' } },
                    y: {}
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    // ─── Section 4: Exposure ranking ─────────────────────────

    var rankCtx = document.getElementById('exposureRankChart');
    if (rankCtx) {
        new Chart(rankCtx, {
            type: 'bar',
            data: {
                labels: ranked.map(function (c) { return c.name; }),
                datasets: [{
                    label: 'Potential New Gas (MW)',
                    data: ranked.map(function (c) { return c.newGasMW; }),
                    backgroundColor: ranked.map(function (c) { return toRGBA(c.color, 0.25); }),
                    borderColor: ranked.map(function (c) { return c.color; }),
                    borderWidth: 2,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: 'New Gas Capacity (MW)' } },
                    y: {}
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    // ─── Section 5: Counterfactual (PPA level proxy) ─────────

    var cfCtx = document.getElementById('counterfactualChart');
    if (cfCtx) {
        var sorted = companyGasData.slice().sort(function (a, b) { return b.ppaDelta - a.ppaDelta; });
        new Chart(cfCtx, {
            type: 'bar',
            data: {
                labels: sorted.map(function (c) { return c.name; }),
                datasets: [{
                    label: 'Low PPA Depth (2040 Mt)',
                    data: sorted.map(function (c) { return c.lowPPA_2040 !== null ? +c.lowPPA_2040.toFixed(2) : 0; }),
                    backgroundColor: toRGBA('#EF4444', 0.25),
                    borderColor: '#EF4444',
                    borderWidth: 1.5,
                    borderRadius: 4
                }, {
                    label: 'High PPA Depth (2040 Mt)',
                    data: sorted.map(function (c) { return c.highPPA_2040 !== null ? +c.highPPA_2040.toFixed(2) : 0; }),
                    backgroundColor: toRGBA('#22C55E', 0.25),
                    borderColor: '#22C55E',
                    borderWidth: 1.5,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { title: { display: true, text: 'CO₂ Emissions at 2040 (Mt)' } },
                    x: { grid: { display: false } }
                },
                plugins: {
                    legend: { position: 'bottom', labels: { usePointStyle: true, padding: 12 } }
                }
            }
        });
    }

})();
