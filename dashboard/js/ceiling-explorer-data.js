// ============================================================================
// ceiling-explorer-metrics.js
// Pure functions for derived metrics — no DOM, no Chart.js, no side effects.
// Depends only on: window.CeilingExplorerData
// ============================================================================

var CeilingMetrics = (function() {
    'use strict';

    var D = function() { return window.CeilingExplorerData; };

    // -- Helpers --------------------------------------------------------------

    /** Get demand_twh for an ISO (needed for marginal LCOE). */
    function getDemandTWh(iso) {
        var d = D();
        var idx = d.meta.isos.indexOf(iso);
        if (idx < 0) return null;
        return d.capacity.demand_twh[idx];
    }

    /** Get a cost curve array by ISO and key (e.g. 'onshore_only_2025_Medium'). */
    function getCurve(iso, curveKey) {
        var d = D();
        if (!d.cost_curves[iso]) return null;
        return d.cost_curves[iso][curveKey] || null;
    }

    /** Filter to points with non-null LCOE. */
    function withLCOE(points) {
        return points.filter(function(p) { return p.lcoe !== null && p.lcoe !== undefined; });
    }

    // -- Core Metrics ---------------------------------------------------------

    /**
     * computeMarginalLCOE(points, demandTWh)
     *
     * Returns array of {score, marginal_lcoe, raw_marginal} where:
     *   - score = midpoint of the step
     *   - raw_marginal = unsmoothed marginal cost
     *   - marginal_lcoe = 3-point centered moving average
     *
     * Formula per step:
     *   marginal = (Δcost_B × 1e9) / ((Δscore/100) × demandTWh × 1e6)
     *            = $/MWh cost of the incremental clean energy in that step
     */
    function computeMarginalLCOE(points, demandTWh) {
        if (!points || points.length < 2 || !demandTWh) return [];

        var raws = [];
        for (var i = 1; i < points.length; i++) {
            var ds = points[i].score - points[i - 1].score;
            var dc = points[i].cost_B - points[i - 1].cost_B;
            if (ds <= 0) continue;
            var marginal = (dc * 1e9) / ((ds / 100) * demandTWh * 1e6);
            raws.push({
                score: (points[i - 1].score + points[i].score) / 2,
                score_end: points[i].score,
                raw_marginal: marginal
            });
        }

        // 3-point centered moving average for smoothing
        var result = [];
        for (var j = 0; j < raws.length; j++) {
            var start = Math.max(0, j - 1);
            var end = Math.min(raws.length, j + 2);
            var sum = 0;
            for (var k = start; k < end; k++) sum += raws[k].raw_marginal;
            result.push({
                score: raws[j].score,
                score_end: raws[j].score_end,
                marginal_lcoe: sum / (end - start),
                raw_marginal: raws[j].raw_marginal
            });
        }
        return result;
    }

    /**
     * getCheapStretchAvg(iso, curveKey)
     *
     * Average LCOE for all points with score < 75% and non-null LCOE.
     * Returns the $/MWh baseline cost of cheap renewable energy.
     */
    function getCheapStretchAvg(iso, curveKey) {
        curveKey = curveKey || 'onshore_only_2025_Medium';
        var curve = getCurve(iso, curveKey);
        if (!curve) return null;

        var pts = withLCOE(curve).filter(function(p) { return p.score < 75; });
        if (pts.length === 0) return null;

        var sum = 0;
        for (var i = 0; i < pts.length; i++) sum += pts[i].lcoe;
        return sum / pts.length;
    }

    /**
     * findElbow(iso, curveKey)
     *
     * Returns {score, lcoe} where average LCOE first exceeds 2× the
     * cheap-stretch average. The "elbow" = where costs start bending sharply.
     */
    function findElbow(iso, curveKey) {
        curveKey = curveKey || 'onshore_only_2025_Medium';
        var avg = getCheapStretchAvg(iso, curveKey);
        if (avg === null) return null;

        var curve = getCurve(iso, curveKey);
        var pts = withLCOE(curve);
        var threshold = 2 * avg;

        for (var i = 0; i < pts.length; i++) {
            if (pts[i].lcoe > threshold) {
                return { score: pts[i].score, lcoe: pts[i].lcoe, threshold: threshold };
            }
        }
        return null; // never exceeds 2×
    }

    /**
     * findCrossover(iso, curveKey, refCost)
     *
     * Returns the CFE score where marginal LCOE permanently exceeds refCost.
     * Uses "last dip below" approach: finds the highest score where smoothed
     * marginal LCOE is still ≤ refCost, then interpolates to the crossing.
     * This avoids false early crossovers caused by noisy discrete capacity steps.
     *
     * refCost benchmarks:
     *   65  = fossil gas
     *   105 = clean firm (nuclear/CCS)
     *   150 = battery 4hr effective
     *   200 = LDES effective
     */
    function findCrossover(iso, curveKey, refCost) {
        curveKey = curveKey || 'onshore_only_2025_Medium';
        var curve = getCurve(iso, curveKey);
        var demand = getDemandTWh(iso);
        if (!curve || !demand) return null;

        var marginals = computeMarginalLCOE(curve, demand);
        if (marginals.length === 0) return null;

        // Find the LAST index where marginal ≤ refCost
        var lastBelow = -1;
        for (var i = 0; i < marginals.length; i++) {
            if (marginals[i].marginal_lcoe <= refCost) {
                lastBelow = i;
            }
        }

        if (lastBelow < 0) {
            // Marginal always exceeds refCost — crossover is at the start
            return { score: marginals[0].score, marginal_lcoe: marginals[0].marginal_lcoe };
        }
        if (lastBelow === marginals.length - 1) {
            return null; // never permanently crosses
        }

        // Interpolate between lastBelow and lastBelow+1
        var prev = marginals[lastBelow];
        var curr = marginals[lastBelow + 1];
        var frac = (refCost - prev.marginal_lcoe) / (curr.marginal_lcoe - prev.marginal_lcoe);
        var interpScore = prev.score + frac * (curr.score - prev.score);
        return { score: interpScore, marginal_lcoe: refCost };
    }

    /**
     * getOffshoreReduction(iso, targetScore)
     *
     * At a target CFE score, compares average LCOE between onshore-only and
     * onshore+offshore cost curves (2025 Medium). Returns % cost reduction.
     */
    function getOffshoreReduction(iso, targetScore) {
        targetScore = targetScore || 95;
        var d = D();
        if (d.meta.offshore_isos.indexOf(iso) < 0) return null;

        var onCurve = getCurve(iso, 'onshore_only_2025_Medium');
        var offCurve = getCurve(iso, 'onshore_plus_offshore_2025_Medium');
        if (!onCurve || !offCurve) return null;

        function nearest(curve, target) {
            var best = null, bestDist = Infinity;
            for (var i = 0; i < curve.length; i++) {
                var dist = Math.abs(curve[i].score - target);
                if (dist < bestDist) { bestDist = dist; best = curve[i]; }
            }
            return best;
        }

        var onPt = nearest(withLCOE(onCurve), targetScore);
        var offPt = nearest(withLCOE(offCurve), targetScore);
        if (!onPt || !offPt) return null;

        var reduction = (onPt.lcoe - offPt.lcoe) / onPt.lcoe * 100;
        return {
            pct_reduction: reduction,
            onshore_lcoe: onPt.lcoe,
            offshore_lcoe: offPt.lcoe,
            onshore_score: onPt.score,
            offshore_score: offPt.score
        };
    }

    /**
     * getSweetSpot(iso, curveKey)
     *
     * The highest CFE score where smoothed marginal LCOE ≤ $150/MWh
     * (battery 4hr benchmark). This is the economic ceiling for renewables-only.
     */
    function getSweetSpot(iso, curveKey) {
        curveKey = curveKey || 'onshore_only_2025_Medium';
        var curve = getCurve(iso, curveKey);
        var demand = getDemandTWh(iso);
        if (!curve || !demand) return null;

        var marginals = computeMarginalLCOE(curve, demand);
        var BATTERY_REF = 150;
        var best = null;

        for (var i = 0; i < marginals.length; i++) {
            if (marginals[i].marginal_lcoe <= BATTERY_REF) {
                // Find the matching cost_curve point for curtailment + avg LCOE
                var pt = null;
                for (var j = 0; j < curve.length; j++) {
                    if (curve[j].score >= marginals[i].score_end - 0.01) {
                        pt = curve[j];
                        break;
                    }
                }
                best = {
                    score: marginals[i].score_end,
                    marginal_lcoe: marginals[i].marginal_lcoe,
                    avg_lcoe: pt ? pt.lcoe : null,
                    curt_pct: pt ? pt.curt_pct : null
                };
            }
        }
        return best;
    }

    /**
     * getSummaryRow(iso)
     *
     * Returns all derived metrics for one ISO, for the summary table.
     */
    function getSummaryRow(iso) {
        var key = 'onshore_only_2025_Medium';
        var d = D();
        var baseIdx = d.meta.isos.indexOf(iso);
        var baseline = d.headline.baseline_score_pct[baseIdx];
        var cheapAvg = getCheapStretchAvg(iso, key);
        var elbow = findElbow(iso, key);
        var xGas = findCrossover(iso, key, 65);
        var xFirm = findCrossover(iso, key, 105);
        var xBatt = findCrossover(iso, key, 150);
        var xLDES = findCrossover(iso, key, 200);
        var sweet = getSweetSpot(iso, key);

        // Offshore crossover (battery) if available
        var xBattOff = null;
        if (d.meta.offshore_isos.indexOf(iso) >= 0) {
            xBattOff = findCrossover(iso, 'onshore_plus_offshore_2025_Medium', 150);
        }
        var offRed = getOffshoreReduction(iso, 95);

        return {
            iso: iso,
            baseline_pct: baseline,
            cheap_avg_lcoe: cheapAvg,
            elbow_score: elbow ? elbow.score : null,
            crossover_gas: xGas ? xGas.score : null,
            crossover_firm: xFirm ? xFirm.score : null,
            crossover_battery: xBatt ? xBatt.score : null,
            crossover_ldes: xLDES ? xLDES.score : null,
            sweet_spot: sweet,
            crossover_battery_offshore: xBattOff ? xBattOff.score : null,
            offshore_reduction_95: offRed ? offRed.pct_reduction : null
        };
    }

    /**
     * getAllSummaryRows()
     *
     * Summary table data for all ISOs.
     */
    function getAllSummaryRows() {
        var d = D();
        return d.meta.isos.map(function(iso) { return getSummaryRow(iso); });
    }

    // -- Public API -----------------------------------------------------------
    return {
        computeMarginalLCOE: computeMarginalLCOE,
        getCheapStretchAvg: getCheapStretchAvg,
        findElbow: findElbow,
        findCrossover: findCrossover,
        getOffshoreReduction: getOffshoreReduction,
        getSweetSpot: getSweetSpot,
        getSummaryRow: getSummaryRow,
        getAllSummaryRows: getAllSummaryRows,
        getDemandTWh: getDemandTWh,
        getCurve: getCurve
    };
})();
