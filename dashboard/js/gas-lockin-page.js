/**
 * Gas Lock-In & Buyer Influence - Cross-Company Analysis
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

    var data = IPP_SMARTARGETS_DATA;
    var companyIds = Object.keys(data.companies);

    var COMPANY_COLORS = [
        '#6366F1', '#22C55E', '#F59E0B', '#E91E63', '#0EA5E9', '#9C27B0', '#14B8A6'
    ];

    // Gas development drivers by ISO (same as ipp-company-report.js)
    var ISO_DRIVERS = {
        ERCOT: { pct: 0.15 }, PJM: { pct: 0.10 }, CAISO: { pct: 0.05 },
        NYISO: { pct: 0 }, NEISO: { pct: 0.08 }, MISO: { pct: 0.10 }, SPP: { pct: 0 }
    };

    function fmt(n, d) {
        if (n === undefined || n === null) return '\u2014';
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

    // ─── Compute gas development data per company ──────────────────

    var companyGasData = companyIds.map(function (id, idx) {
        var co = data.companies[id];
        var fleet = co.fleet_summary || {};
        var gasCap = 0;
        var nuclearCap = 0;
        var gasISOs = {};

        for (var iso in fleet) {
            var isoGas = (fleet[iso].gas_ccgt ? fleet[iso].gas_ccgt.cap_mw : 0) +
                         (fleet[iso].gas_peaker ? fleet[iso].gas_peaker.cap_mw : 0);
            var isoNuclear = fleet[iso].nuclear ? fleet[iso].nuclear.cap_mw : 0;
            if (isoGas > 0) {
                gasISOs[iso] = isoGas;
                gasCap += isoGas;
            }
            nuclearCap += isoNuclear;
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
        var cumulativeCO2 = annualCO2 * (1.0 + 0.9 + 0.7 + 0.5 + 0.3) * 5;

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
            nuclearCap: nuclearCap,
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

    // Aggregate totals
    var totalNewGW = companyGasData.reduce(function (s, c) { return s + c.newGasMW; }, 0) / 1000;
    var totalCumCO2 = companyGasData.reduce(function (s, c) { return s + c.cumulativeCO2; }, 0);
    var totalAnnualCO2 = companyGasData.reduce(function (s, c) { return s + c.annualCO2; }, 0);
    var strandedValue = totalNewGW * 1200;
    var totalNuclearGW = companyGasData.reduce(function (s, c) { return s + c.nuclearCap; }, 0) / 1000;

    // ISO-level totals for narrative
    var allISOs = ['ERCOT', 'PJM', 'MISO', 'NEISO', 'CAISO'];
    var isoTotals = {};
    allISOs.forEach(function (iso) {
        isoTotals[iso] = companyGasData.reduce(function (s, c) { return s + (c.newGasByISO[iso] || 0); }, 0);
    });
    var topISO = allISOs.slice().sort(function (a, b) { return (isoTotals[b] || 0) - (isoTotals[a] || 0); });

    // PPA counterfactual totals
    var totalPPADelta = companyGasData.reduce(function (s, c) { return s + c.ppaDelta; }, 0);
    var sortedByDelta = companyGasData.slice().sort(function (a, b) { return b.ppaDelta - a.ppaDelta; });

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
                    backgroundColor: toRGBA('#EF4444', 0.25),
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
        var isoData = allISOs.map(function (iso) { return isoTotals[iso]; });

        new Chart(isoCtx, {
            type: 'bar',
            data: {
                labels: allISOs,
                datasets: [{
                    label: 'Total New Gas (MW)',
                    data: isoData,
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
                    y: { title: { display: true, text: 'New Gas (MW)' }, beginAtZero: true },
                    x: { grid: { display: false } }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    // Section 2 narrative injection
    var exposureNarr = document.getElementById('exposureNarrative');
    if (exposureNarr) {
        var top2 = ranked.slice(0, 2);
        exposureNarr.innerHTML =
            '<p>Under market-driven conditions (Strategy 1B), the 7 IPPs collectively develop <strong>' + fmt(totalNewGW, 1) +
            ' GW</strong> of new gas-fired capacity. ' + top2[0].name + ' and ' + top2[1].name +
            ' account for the largest share, with ' + fmt(top2[0].newGasMW) + ' MW and ' + fmt(top2[1].newGasMW) +
            ' MW respectively, driven by their existing gas-heavy fleets and ISO footprints in regions where capacity market revenue and scarcity pricing make gas investment rational.</p>' +
            '<p>The development calculus is straightforward: gas plants earn their primary revenue from capacity payments and scarcity hours, not wholesale energy. ' +
            'In ISOs with energy-only markets (ERCOT) or deep capacity markets (PJM), the financial case for new gas persists even as average wholesale prices decline. ' +
            '<strong>This is the lock-in vector: each GW of gas built creates a 30-40 year commitment that reshapes the political economy of decarbonization.</strong></p>';
    }

    var isoNarr = document.getElementById('isoNarrative');
    if (isoNarr) {
        isoNarr.innerHTML =
            '<p>New gas development concentrates in ' + topISO[0] + ' (' + fmt(isoTotals[topISO[0]]) +
            ' MW) and ' + topISO[1] + ' (' + fmt(isoTotals[topISO[1]]) + ' MW). ' +
            topISO[0] + ' leads because its energy-only market design creates volatile scarcity pricing that rewards gas peakers. ' +
            topISO[1] + '\'s deep capacity market provides reliable revenue streams that underwrite new gas construction.</p>' +
            '<p>ISOs with lower gas development (' + topISO[topISO.length - 1] + ', ' + topISO[topISO.length - 2] +
            ') either have strong clean energy mandates or limited gas fleet footprints among these 7 IPPs. ' +
            '<strong>The regional pattern matters: gas lock-in is not uniform across the U.S. grid. It concentrates where market design rewards dispatchable capacity, regardless of carbon intensity.</strong></p>';
    }

    // ─── Section 3: Lock-in cost stats + cumulative emissions ─────

    var statsEl = document.getElementById('lockinStats');
    if (statsEl) {
        statsEl.innerHTML = '<div class="stat-card"><div class="stat-value">' + fmt(totalNewGW, 1) + ' GW</div><div class="stat-label">Total New Gas at Risk</div></div>' +
            '<div class="stat-card"><div class="stat-value">' + fmt(totalAnnualCO2, 1) + ' Mt/yr</div><div class="stat-label">Annual CO\u2082 from New Gas</div></div>' +
            '<div class="stat-card"><div class="stat-value">' + fmt(totalCumCO2, 0) + ' Mt</div><div class="stat-label">Cumulative CO\u2082 through 2050</div></div>' +
            '<div class="stat-card"><div class="stat-value">$' + fmt(strandedValue, 0) + 'M</div><div class="stat-label">Stranded Asset Value</div></div>';
    }

    var cumCtx = document.getElementById('cumulativeEmissionsChart');
    if (cumCtx) {
        new Chart(cumCtx, {
            type: 'bar',
            data: {
                labels: ranked.map(function (c) { return c.name; }),
                datasets: [{
                    label: 'Cumulative Locked-In CO\u2082 (Mt)',
                    data: ranked.map(function (c) { return +c.cumulativeCO2.toFixed(1); }),
                    backgroundColor: toRGBA('#EF4444', 0.25),
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
                    x: { title: { display: true, text: 'Mt CO\u2082 (2025\u20132050)' } },
                    y: {}
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    // Section 3 narrative
    var lockinNarr = document.getElementById('lockinCostNarrative');
    if (lockinNarr) {
        var carbonPrice50 = totalCumCO2 * 50;   // $M at $50/ton
        var carbonPrice100 = totalCumCO2 * 100;  // $M at $100/ton
        lockinNarr.innerHTML =
            '<p>If all projected gas gets built, these 7 IPPs lock in <strong>' + fmt(totalCumCO2, 0) +
            ' Mt of cumulative CO\u2082</strong> through 2050, assuming declining dispatch as clean energy scales (full utilization years 1\u20135, declining to 30% by year 25). ' +
            'At today\'s new-build cost of ~$1.2M/MW, the total capital at risk is <strong>$' + fmt(strandedValue, 0) + 'M</strong>.</p>' +
            '<p>The stranding math sharpens with carbon pricing. At $50/ton (EPA social cost of carbon), the emissions liability alone reaches <strong>$' + fmt(carbonPrice50 / 1000, 1) +
            'B</strong>. At $100/ton (approaching EU ETS levels), it climbs to <strong>$' + fmt(carbonPrice100 / 1000, 1) +
            'B</strong>. These liabilities compound: gas plants built in 2025\u20132030 will operate for 30\u201340 years, locking in emissions through 2060+ unless retired early at a loss.</p>' +
            '<p><strong>This is the asymmetry of gas lock-in: the capital commitment is immediate but the emissions liability stretches decades, growing more costly as carbon pricing tightens.</strong> ' +
            'For company-level stranding risk analysis, see the individual <a href="ipp_climate_transition.html" class="explore-link" style="font-size:inherit;">IPP assessments</a>.</p>';
    }

    // ─── Section 4: What Replaces Gas ─────────────────────────

    var replCtx = document.getElementById('replacementChart');
    if (replCtx) {
        // Show existing clean firm (nuclear) capacity vs new gas at risk per company
        var replSorted = companyGasData.slice().sort(function (a, b) { return b.newGasMW - a.newGasMW; });

        new Chart(replCtx, {
            type: 'bar',
            data: {
                labels: replSorted.map(function (c) { return c.name; }),
                datasets: [{
                    label: 'Existing Nuclear (MW)',
                    data: replSorted.map(function (c) { return c.nuclearCap; }),
                    backgroundColor: toRGBA(RESOURCE_COLORS.nuclear || '#6366F1', 0.25),
                    borderColor: RESOURCE_COLORS.nuclear || '#6366F1',
                    borderWidth: 1.5,
                    borderRadius: 4
                }, {
                    label: 'New Gas at Risk (MW)',
                    data: replSorted.map(function (c) { return c.newGasMW; }),
                    backgroundColor: toRGBA('#EF4444', 0.25),
                    borderColor: '#EF4444',
                    borderWidth: 1.5,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { title: { display: true, text: 'Capacity (MW)' }, beginAtZero: true },
                    x: { grid: { display: false } }
                },
                plugins: {
                    legend: { position: 'bottom', labels: { usePointStyle: true, padding: 12 } }
                }
            }
        });
    }

    // Section 4 narrative
    var replNarr = document.getElementById('replacementNarrative');
    if (replNarr) {
        // Find companies with most nuclear
        var nuclearSorted = companyGasData.slice().sort(function (a, b) { return b.nuclearCap - a.nuclearCap; });
        var topNuc = nuclearSorted.filter(function (c) { return c.nuclearCap > 0; });

        replNarr.innerHTML =
            '<p>The clean firm technologies that displace gas under hourly matching are nuclear (existing fleet + license extensions), CCS-CCGT, and long-duration energy storage (LDES). ' +
            'Across the 7 IPPs, <strong>' + fmt(totalNuclearGW, 1) + ' GW</strong> of existing nuclear capacity is the foundation of any gas displacement strategy\u2014already dispatchable, zero-carbon, and fully amortized.</p>' +
            '<p>' + (topNuc.length > 0 ? topNuc[0].name + ' leads with ' + fmt(topNuc[0].nuclearCap) + ' MW of nuclear capacity, ' +
            (topNuc.length > 1 ? 'followed by ' + topNuc[1].name + ' (' + fmt(topNuc[1].nuclearCap) + ' MW). ' : '. ') : '') +
            'Companies with large nuclear fleets are best positioned to transition: their existing clean firm generation covers a substantial share of the hours that new gas would serve. ' +
            'The gap\u2014particularly overnight and seasonal shortfalls\u2014requires new CCS-CCGT and LDES capacity, technologies currently at FOAK cost but trending toward parity via Wright\'s Law learning curves.</p>' +
            '<!-- QA FLAG: replacement chart shows nuclear only; CCS-CCGT and LDES capacity data not available per-company from IPP_SMARTARGETS_DATA. True replacement mix requires pipeline step computing technology-specific displacement under 2C strategy. -->' +
            '<p><strong>The investment question isn\'t whether clean firm can replace gas\u2014the technology portfolio exists. It\'s whether procurement signals arrive fast enough to make the FOAK-to-NOAK cost transition before the next wave of gas gets locked in.</strong> ' +
            'See the <a href="clean_firm_case.html" class="explore-link" style="font-size:inherit;">Clean Firm Investment Case</a> for the full critical mass and learning curve analysis.</p>';
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
                    y: { title: { display: true, text: 'CO\u2082 Emissions at 2040 (Mt)' }, beginAtZero: true },
                    x: { grid: { display: false } }
                },
                plugins: {
                    legend: { position: 'bottom', labels: { usePointStyle: true, padding: 12 } }
                }
            }
        });
    }

    // Section 5 narrative
    var cfNarr = document.getElementById('counterfactualNarrative');
    if (cfNarr) {
        var topDelta = sortedByDelta.slice(0, 3);
        cfNarr.innerHTML =
            '<p>The chart above compares each company\'s projected 2040 emissions under low vs. high PPA market depth\u2014a proxy for the difference between consequential netting (Strategy 1B) and hourly matching (Strategy 2C). ' +
            'Across all 7 IPPs, high PPA depth reduces aggregate 2040 emissions by <strong>' + fmt(totalPPADelta, 1) + ' Mt CO\u2082</strong>.</p>' +
            '<p>' + topDelta[0].name + ' shows the largest swing (' + fmt(topDelta[0].ppaDelta, 2) + ' Mt reduction), ' +
            (topDelta.length > 1 ? 'followed by ' + topDelta[1].name + ' (' + fmt(topDelta[1].ppaDelta, 2) + ' Mt) and ' +
            topDelta[2].name + ' (' + fmt(topDelta[2].ppaDelta, 2) + ' Mt). ' : '. ') +
            'Companies with the largest gas fleets in capacity-rich ISOs see the biggest emissions swing, because hourly matching directly displaces the scarcity hours that justify gas construction.</p>' +
            '<p><strong>This is the analytical punchline: the procurement signal buyers send today determines not just which PPAs get signed, but which power plants get built\u2014and the cumulative emissions trajectory of the entire fleet through mid-century.</strong> ' +
            'The gap between low and high PPA depth isn\'t a marginal adjustment. It\'s the difference between locking in gas for decades or building the clean firm portfolio that makes gas unnecessary.</p>';
    }

    // ─── Section 6: Exposure ranking + drill-down intro ─────

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

    // Section 6 drill-down narrative
    var drillNarr = document.getElementById('drilldownNarrative');
    if (drillNarr) {
        drillNarr.innerHTML =
            '<p>Each IPP\'s gas lock-in risk depends on three factors: existing fleet composition (how much gas they already operate), ISO footprint (which markets reward gas development), and competitive position (whether they have clean firm alternatives ready to deploy). ' +
            'The ranking below shows each company\'s potential new gas capacity under market-driven conditions. Click through to explore the full gas development scenario for each company, including ISO-specific drivers, emissions trajectories, and the strategic fork between gas expansion and clean firm investment.</p>';
    }

})();
