// ============================================================================
// CEILING EXPLORER METRICS — DEVTOOLS TEST HARNESS
// ============================================================================
// Paste this into the browser console after loading ceiling_explorer.html
// (which includes ceiling-explorer-data.js and ceiling-explorer-metrics.js).
//
// It validates every derived metric against expected behavior and prints
// a pass/fail table.
// ============================================================================

(function() {
    'use strict';

    var M = CeilingMetrics;
    var D = window.CeilingExplorerData;
    var pass = 0, fail = 0, results = [];

    function assert(name, condition, detail) {
        if (condition) {
            pass++;
            results.push({ test: name, status: '✅ PASS', detail: detail || '' });
        } else {
            fail++;
            results.push({ test: name, status: '❌ FAIL', detail: detail || '' });
        }
    }

    // ---- 1. getDemandTWh ----------------------------------------------------
    var ercotDemand = M.getDemandTWh('ERCOT');
    assert('getDemandTWh ERCOT', ercotDemand === 488.02,
        'Expected 488.02, got ' + ercotDemand);

    assert('getDemandTWh unknown ISO', M.getDemandTWh('FAKE') === null,
        'Should return null for unknown ISO');

    // ---- 2. getCurve --------------------------------------------------------
    var curve = M.getCurve('ERCOT', 'onshore_only_2025_Medium');
    assert('getCurve returns array', Array.isArray(curve) && curve.length > 20,
        'Length: ' + (curve ? curve.length : 'null'));

    assert('getCurve first point has null lcoe', curve[0].lcoe === null,
        'First point lcoe: ' + curve[0].lcoe);

    assert('getCurve missing key returns null',
        M.getCurve('ERCOT', 'nonexistent_key') === null);

    // ---- 3. getCheapStretchAvg ----------------------------------------------
    var cheapERCOT = M.getCheapStretchAvg('ERCOT');
    assert('cheapStretchAvg ERCOT in range',
        cheapERCOT !== null && cheapERCOT > 40 && cheapERCOT < 70,
        '$' + (cheapERCOT ? cheapERCOT.toFixed(1) : 'null') + '/MWh');

    var cheapNYISO = M.getCheapStretchAvg('NYISO');
    assert('cheapStretchAvg NYISO > ERCOT',
        cheapNYISO > cheapERCOT,
        'NYISO=$' + (cheapNYISO ? cheapNYISO.toFixed(1) : '?') +
        ' vs ERCOT=$' + (cheapERCOT ? cheapERCOT.toFixed(1) : '?'));

    // ---- 4. computeMarginalLCOE ---------------------------------------------
    var marginals = M.computeMarginalLCOE(curve, ercotDemand);
    assert('marginalLCOE returns array', Array.isArray(marginals) && marginals.length > 10,
        'Length: ' + marginals.length);

    assert('marginalLCOE has score and marginal_lcoe fields',
        marginals[0].score !== undefined && marginals[0].marginal_lcoe !== undefined);

    // Marginal should increase monotonically in the tail (last 5 points)
    var tail = marginals.slice(-5);
    var monotonic = true;
    for (var i = 1; i < tail.length; i++) {
        if (tail[i].marginal_lcoe < tail[i-1].marginal_lcoe) monotonic = false;
    }
    assert('marginalLCOE tail is monotonically increasing', monotonic,
        'Last 5: ' + tail.map(function(m) { return '$' + m.marginal_lcoe.toFixed(0); }).join(', '));

    // Last marginal should be very high (>$1000)
    var lastMarg = marginals[marginals.length - 1];
    assert('marginalLCOE final point > $1000',
        lastMarg.marginal_lcoe > 1000,
        'Final: $' + lastMarg.marginal_lcoe.toFixed(0) + '/MWh at ' +
        lastMarg.score.toFixed(1) + '%');

    // ---- 5. findElbow -------------------------------------------------------
    var elbowERCOT = M.findElbow('ERCOT');
    assert('findElbow ERCOT returns object',
        elbowERCOT !== null && elbowERCOT.score !== undefined,
        'Score: ' + (elbowERCOT ? elbowERCOT.score.toFixed(1) + '%, LCOE: $' +
        elbowERCOT.lcoe.toFixed(0) : 'null'));

    assert('findElbow ERCOT in 85-98 range',
        elbowERCOT && elbowERCOT.score > 85 && elbowERCOT.score < 98,
        'Score: ' + (elbowERCOT ? elbowERCOT.score.toFixed(1) : '?'));

    // ---- 6. findCrossover ---------------------------------------------------
    var xGas = M.findCrossover('ERCOT', null, 65);
    var xFirm = M.findCrossover('ERCOT', null, 105);
    var xBatt = M.findCrossover('ERCOT', null, 150);
    var xLDES = M.findCrossover('ERCOT', null, 200);

    assert('crossover gas < firm < battery < LDES (ERCOT)',
        xGas && xFirm && xBatt && xLDES &&
        xGas.score < xFirm.score && xFirm.score < xBatt.score && xBatt.score < xLDES.score,
        'Gas=' + (xGas ? xGas.score.toFixed(1) : '?') +
        ' Firm=' + (xFirm ? xFirm.score.toFixed(1) : '?') +
        ' Batt=' + (xBatt ? xBatt.score.toFixed(1) : '?') +
        ' LDES=' + (xLDES ? xLDES.score.toFixed(1) : '?'));

    // NYISO should cross battery at a lower score than ERCOT
    var xBattNY = M.findCrossover('NYISO', null, 150);
    assert('NYISO battery crossover < ERCOT battery crossover',
        xBattNY && xBatt && xBattNY.score < xBatt.score,
        'NYISO=' + (xBattNY ? xBattNY.score.toFixed(1) : '?') +
        ' ERCOT=' + (xBatt ? xBatt.score.toFixed(1) : '?'));

    // ---- 7. getOffshoreReduction --------------------------------------------
    var offNY = M.getOffshoreReduction('NYISO', 95);
    assert('offshore NYISO reduction > 30%',
        offNY !== null && offNY.pct_reduction > 30,
        offNY ? offNY.pct_reduction.toFixed(1) + '% reduction' : 'null');

    var offCA = M.getOffshoreReduction('CAISO', 95);
    assert('offshore CAISO reduction < 15%',
        offCA !== null && offCA.pct_reduction < 15,
        offCA ? offCA.pct_reduction.toFixed(1) + '% reduction' : 'null');

    assert('offshore non-coastal ISO returns null',
        M.getOffshoreReduction('ERCOT', 95) === null);

    // ---- 8. getSweetSpot ----------------------------------------------------
    var sweetERCOT = M.getSweetSpot('ERCOT');
    assert('sweetSpot ERCOT returns object',
        sweetERCOT !== null && sweetERCOT.score !== undefined,
        sweetERCOT ? 'Score=' + sweetERCOT.score.toFixed(1) +
        '% LCOE=$' + (sweetERCOT.avg_lcoe ? sweetERCOT.avg_lcoe.toFixed(0) : '?') +
        ' Curt=' + (sweetERCOT.curt_pct ? sweetERCOT.curt_pct.toFixed(0) : '?') + '%'
        : 'null');

    assert('sweetSpot score > 80',
        sweetERCOT && sweetERCOT.score > 80);

    // ---- 9. getSummaryRow ---------------------------------------------------
    var row = M.getSummaryRow('ERCOT');
    assert('summaryRow has all fields',
        row && row.iso === 'ERCOT' &&
        row.baseline_pct !== undefined &&
        row.cheap_avg_lcoe !== null &&
        row.elbow_score !== null &&
        row.crossover_battery !== null &&
        row.sweet_spot !== null,
        JSON.stringify(row, null, 0).substring(0, 200));

    // ---- 10. getAllSummaryRows -----------------------------------------------
    var allRows = M.getAllSummaryRows();
    assert('getAllSummaryRows returns 7 rows', allRows.length === 7);

    // Check that offshore-only ISOs have offshore data
    var offshoreIsos = D.meta.offshore_isos;
    allRows.forEach(function(r) {
        if (offshoreIsos.indexOf(r.iso) >= 0) {
            assert(r.iso + ' has offshore crossover',
                r.crossover_battery_offshore !== null,
                'Offshore battery crossover: ' +
                (r.crossover_battery_offshore ? r.crossover_battery_offshore.toFixed(1) : 'null'));
        } else {
            assert(r.iso + ' has no offshore data',
                r.crossover_battery_offshore === null);
        }
    });

    // ---- Print Summary Table ------------------------------------------------
    console.log('\n========================================');
    console.log('CEILING EXPLORER METRICS — TEST RESULTS');
    console.log('========================================\n');
    console.table(results);
    console.log('\n' + pass + ' passed, ' + fail + ' failed out of ' +
                (pass + fail) + ' tests.\n');

    // ---- Print Summary Data Table -------------------------------------------
    console.log('\n=== SUMMARY TABLE (for punchline section) ===\n');
    var tableRows = allRows.map(function(r) {
        return {
            ISO: r.iso,
            'Baseline %': r.baseline_pct ? r.baseline_pct.toFixed(1) : '-',
            'Cheap $/MWh': r.cheap_avg_lcoe ? '$' + r.cheap_avg_lcoe.toFixed(0) : '-',
            'Elbow %': r.elbow_score ? r.elbow_score.toFixed(1) : '-',
            'Gas xover %': r.crossover_gas ? r.crossover_gas.toFixed(1) : '-',
            'Firm xover %': r.crossover_firm ? r.crossover_firm.toFixed(1) : '-',
            'Batt xover %': r.crossover_battery ? r.crossover_battery.toFixed(1) : '-',
            'Batt+Off %': r.crossover_battery_offshore ?
                r.crossover_battery_offshore.toFixed(1) : '-',
            'Sweet spot %': r.sweet_spot ? r.sweet_spot.score.toFixed(1) : '-',
            'Curt @ sweet': r.sweet_spot && r.sweet_spot.curt_pct ?
                r.sweet_spot.curt_pct.toFixed(0) + '%' : '-',
            'Off red @95': r.offshore_reduction_95 ?
                r.offshore_reduction_95.toFixed(0) + '%' : '-'
        };
    });
    console.table(tableRows);

})();
