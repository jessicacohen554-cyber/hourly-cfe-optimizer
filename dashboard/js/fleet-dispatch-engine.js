// ============================================================================
// FLEET DISPATCH ENGINE — Client-side JS port of scripts/fleet_dispatch.py
// ============================================================================
// Computes fleet-level emissions from per-plant data + parametric sweep results.
// All math runs in-browser: no backend round-trip needed.
//
// Input:  plants[] + sweepData (from sweep_dispatch_data.json)
// Output: { envelope, plant_detail, generation_by_fuel, emissions_by_fuel }
//         matching fleet_scenario_results.json schema
// ============================================================================

var FleetDispatchEngine = (function () {
    'use strict';

    // ── Constants (matching fleet_dispatch.py exactly) ──

    var REFERENCE_HEAT_RATES = {
        gas_ccgt: 7.0,
        gas_ct: 10.5,
        oil_ct: 10.5,
        gas_oil_ct: 10.5
    };

    var EMISSION_FACTORS = {
        gas_ccgt: 0.05306,
        gas_ct: 0.05306,
        oil_ct: 0.07396,
        gas_oil_ct: 0.06351
    };

    // Default CO2 rates (t CO2/MWh) from eGRID 2023 CEG fleet averages.
    // Used as fallback when plant has no co2_rate_t_mwh in fleet config.
    var DEFAULT_CO2_RATES = {
        gas_ccgt: 0.382,   // eGRID avg: 843 lb/MWh
        gas_ct: 0.657,     // eGRID avg: 1449 lb/MWh
        oil_ct: 1.387,     // eGRID avg: 3058 lb/MWh
        gas_oil_ct: 1.506  // eGRID avg: 3321 lb/MWh
    };

    var CCS_MILESTONES = [[0, 0.0], [2, 0.30], [5, 0.70], [8, 1.00]];

    // ── CCS ramp schedule ──

    function ccsRampFraction(year, yearOnline, targetRate) {
        if (targetRate === undefined) targetRate = 0.95;
        if (year < yearOnline) return 0.0;

        var elapsed = year - yearOnline;
        if (elapsed >= CCS_MILESTONES[CCS_MILESTONES.length - 1][0]) {
            return targetRate;
        }

        for (var i = 0; i < CCS_MILESTONES.length - 1; i++) {
            var t0 = CCS_MILESTONES[i][0], f0 = CCS_MILESTONES[i][1];
            var t1 = CCS_MILESTONES[i + 1][0], f1 = CCS_MILESTONES[i + 1][1];
            if (elapsed >= t0 && elapsed < t1) {
                var frac = f0 + (f1 - f0) * (elapsed - t0) / (t1 - t0);
                return frac * targetRate;
            }
        }
        return targetRate;
    }

    // ── Percentile helper ──

    function percentile(arr, p) {
        if (arr.length === 0) return 0;
        var sorted = arr.slice().sort(function (a, b) { return a - b; });
        var idx = (p / 100) * (sorted.length - 1);
        var lo = Math.floor(idx);
        var hi = Math.ceil(idx);
        if (lo === hi) return sorted[lo];
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
    }

    // ── Apply scenario modifications to base fleet ──

    function applyScenario(baseFleet, modifications) {
        // Deep clone
        var fleet = JSON.parse(JSON.stringify(baseFleet));

        // Index by orispl
        var byOrispl = {};
        fleet.forEach(function (p) {
            if (!byOrispl[p.orispl]) byOrispl[p.orispl] = [];
            byOrispl[p.orispl].push(p);
        });

        (modifications || []).forEach(function (mod) {
            var action = mod.action;

            if (action === 'add_plant') {
                var fuel = mod.fuel_type;
                var hr = mod.heat_rate_mmbtu_mwh || REFERENCE_HEAT_RATES[fuel] || 10.0;
                var co2 = mod.co2_rate_t_mwh;
                if (co2 == null) {
                    var ef = EMISSION_FACTORS[fuel] || 0.05306;
                    co2 = Math.round(hr * ef * 100000) / 100000;
                }
                fleet.push({
                    orispl: 0,
                    name: mod.name,
                    iso: mod.iso,
                    capacity_mw: mod.capacity_mw,
                    fuel_type: fuel,
                    heat_rate_mmbtu_mwh: hr,
                    co2_rate_t_mwh: co2,
                    equity_share: mod.equity_share || 1.0,
                    status: 'operating',
                    ccs_capture_rate: 0,
                    ccs_heat_rate_penalty: 1.0,
                    _action: 'add_plant',
                    _year_online: mod.year_online,
                    _ccs_target_rate: 0
                });
                return;
            }

            // Find target plants
            var targets = [];
            if (mod.orispl != null) {
                targets = byOrispl[mod.orispl] || [];
            } else if (mod.fuel_type) {
                targets = fleet.filter(function (p) { return p.fuel_type === mod.fuel_type; });
            }

            targets.forEach(function (p) {
                if (action === 'retire') {
                    p._action = 'retire';
                    p._year_online = mod.year_online;
                } else if (action === 'ccs_retrofit') {
                    p._action = 'ccs_retrofit';
                    p._year_online = mod.year_online;
                    p._ccs_target_rate = mod.ccs_capture_rate || 0.95;
                    p.ccs_heat_rate_penalty = mod.ccs_heat_rate_penalty || 1.14;
                }
            });
        });

        // Set defaults for unmodified plants
        fleet.forEach(function (p) {
            if (!p._action) p._action = null;
            if (p._year_online === undefined) p._year_online = null;
            if (p._ccs_target_rate === undefined) p._ccs_target_rate = 0;
            if (!p.equity_share) p.equity_share = 1.0;
            if (!p.ccs_capture_rate) p.ccs_capture_rate = 0;
            if (!p.ccs_heat_rate_penalty) p.ccs_heat_rate_penalty = 1.0;
        });

        return fleet;
    }

    // ── Generation estimates for non-fossil plants (TWh, fleet-level) ──
    // These are used to provide static generation for non-dispatchable plants.
    // Per-plant generation is estimated from fleet totals proportional to capacity.
    var NON_FOSSIL_CF = {
        nuclear: 0.92,      // ~92% capacity factor
        geothermal: 0.85,   // ~85% capacity factor (baseload)
        wind: 0.30,         // ~30% capacity factor (intermittent, national avg)
        solar: 0.22,        // ~22% capacity factor (intermittent, national avg)
        hydro: 0.40,        // ~40% capacity factor (Conowingo)
        battery: 0.0,       // Storage, net zero gen
        ldes: 0.0,          // Long duration storage, net zero gen
        pumped_storage: 0.0 // Pumped storage, net zero gen
    };

    // Regional capacity factors by ISO (from EIA 2023 fleet-weighted averages)
    var REGIONAL_CF = {
        solar: {
            CAISO: 0.27, ERCOT: 0.24, PJM: 0.18, NYISO: 0.16,
            NEISO: 0.16, MISO: 0.20, SPP: 0.23
        },
        wind: {
            CAISO: 0.26, ERCOT: 0.35, PJM: 0.28, NYISO: 0.28,
            NEISO: 0.30, MISO: 0.34, SPP: 0.38
        }
    };

    // Static fallback CFs for fossil plants when sweep data is missing (e.g. 2023 baseline).
    // Derived from eGRID 2023: actual generation / (capacity × 8760) for CEG fleet.
    var FOSSIL_STATIC_CF = {
        gas_ccgt: 0.57,    // eGRID 2023 CEG avg (higher than generic 0.44 due to fleet mix)
        gas_ct: 0.08,
        oil_ct: 0.01,
        gas_oil_ct: 0.03,
        coal_steam: 0.50
    };

    // ── Main dispatch computation ──

    function computeFleetDispatch(plants, sweepData, options) {
        options = options || {};
        // CCS parameters from sidebar panel (override per-plant defaults)
        var globalDeratePct = options.ccs_derate_pct != null ? options.ccs_derate_pct : 14;
        var globalCapturePct = options.ccs_capture_rate_pct != null ? options.ccs_capture_rate_pct : 90;
        var globalCcsCfCap = (options.ccs_cf_pct != null ? options.ccs_cf_pct : 85) / 100.0; // max CF for CCS plants

        var nScenarios = sweepData.n_scenarios;
        var nYears = sweepData.n_years;
        var years = sweepData.years;

        // Filter out permanently retired plants with no action
        var activePlants = plants.filter(function (p) {
            return !(p.status === 'retired' && !p._action);
        });

        // Only dispatch fossil-fuel plants via sweep (non-fossil use static CF)
        var fossilFuels = new Set(['gas_ccgt', 'gas_ct', 'coal_steam', 'oil_ct', 'gas_oil_ct']);

        // Initialize fleet emissions and generation arrays: [scenario][year]
        var fleetEmissions = [];  // Mt CO2
        var fleetGeneration = []; // MWh (for intensity calc)
        for (var s = 0; s < nScenarios; s++) {
            fleetEmissions[s] = new Float64Array(nYears);
            fleetGeneration[s] = new Float64Array(nYears);
        }

        // Per-plant results for P50 detail
        var plantResults = []; // [{plant, genArr[s][y], emisArr[s][y], cfArr[s][y], statusPerYear}]

        // Generation and emissions accumulators by fuel: [scenario][year]
        var genByFuel = {};
        var emisByFuel = {};
        ['coal_steam', 'gas_ccgt', 'gas_ct', 'oil_ct', 'gas_oil_ct', 'nuclear', 'geothermal', 'wind', 'solar', 'hydro', 'ccs_ccgt', 'battery', 'ldes'].forEach(function (f) {
            genByFuel[f] = [];
            emisByFuel[f] = [];
            for (var s = 0; s < nScenarios; s++) {
                genByFuel[f][s] = new Float64Array(nYears);
                emisByFuel[f][s] = new Float64Array(nYears);
            }
        });

        // First pass: handle non-fossil plants (static generation, zero emissions)
        activePlants.forEach(function (p) {
            var fuel = p.fuel_type;
            if (fossilFuels.has(fuel)) return; // Handle in second pass

            // Use custom CF if set, then regional CF if available, then national average
            var cf;
            if (p._custom_cf && p._custom_cf > 0) {
                cf = p._custom_cf;
            } else if (REGIONAL_CF[fuel] && REGIONAL_CF[fuel][p.iso]) {
                cf = REGIONAL_CF[fuel][p.iso];
            } else {
                cf = NON_FOSSIL_CF[fuel] || 0;
            }
            if (cf <= 0) return; // Skip storage (net zero gen)

            var capMW = (p.capacity_mw || 0) * (p.equity_share || 1.0);
            var action = p._action;
            var yearOnline = p._year_online;

            var genArr = [];
            var emisArr = [];
            var cfArr = [];
            var statusPerYear = [];

            for (var si = 0; si < nScenarios; si++) {
                genArr[si] = new Float64Array(nYears);
                emisArr[si] = new Float64Array(nYears);
                cfArr[si] = new Float64Array(nYears);
            }

            // Uprate: additional MW from year_online onward at same CF
            var uprateMW = (action === 'uprate' && p._uprate_mw) ? p._uprate_mw * (p.equity_share || 1.0) : 0;

            for (var yi = 0; yi < nYears; yi++) {
                var y = years[yi];

                // Status
                if (action === 'retire' && yearOnline && y >= yearOnline) {
                    statusPerYear[yi] = 'retired';
                } else if (action === 'add_plant' && yearOnline && y < yearOnline) {
                    statusPerYear[yi] = 'not_yet_built';
                } else if (action === 'uprate' && yearOnline && y >= yearOnline) {
                    statusPerYear[yi] = 'uprated';
                } else {
                    statusPerYear[yi] = p.status || 'operating';
                }

                var activeCf = (statusPerYear[yi] === 'retired' || statusPerYear[yi] === 'not_yet_built') ? 0 : cf;
                var effectiveCap = capMW;
                if (statusPerYear[yi] === 'uprated') {
                    effectiveCap = capMW + uprateMW;
                }
                var genMwh = effectiveCap * activeCf * 8760;

                // Same generation across all sweep scenarios (non-fossil isn't affected by fossil cost sweeps)
                for (var si2 = 0; si2 < nScenarios; si2++) {
                    genArr[si2][yi] = genMwh;
                    emisArr[si2][yi] = 0; // Zero emissions
                    cfArr[si2][yi] = activeCf;
                    fleetGeneration[si2][yi] += genMwh; // Accumulate total gen for intensity

                    if (genByFuel[fuel]) {
                        genByFuel[fuel][si2][yi] += genMwh / 1e6; // TWh
                    }
                }
            }

            plantResults.push({
                plant: p,
                genArr: genArr,
                emisArr: emisArr,
                cfArr: cfArr,
                statusPerYear: statusPerYear
            });
        });

        // Second pass: fossil plants via sweep dispatch
        activePlants.forEach(function (p) {
            var fuel = p.fuel_type;
            if (!fossilFuels.has(fuel)) {
                return; // Already handled above
            }

            var iso = p.iso;
            var isoData = sweepData.data[iso];
            if (!isoData) return;

            var cfKey = fuel + '_cf';
            var marginKey = fuel + '_margin';
            var cfArrays = isoData[cfKey];    // [scenario][year_idx]
            var marginArrays = isoData[marginKey];
            if (!cfArrays || !marginArrays) return;

            // Pre-detect years where ALL sweep CFs are zero (e.g. 2023 baseline)
            // For these years, use static fallback CFs
            var fallbackCf = FOSSIL_STATIC_CF[fuel] || 0.10;
            var yearHasData = new Uint8Array(nYears);
            for (var yCheck = 0; yCheck < nYears; yCheck++) {
                // Check first few scenarios for non-zero data
                for (var sCheck = 0; sCheck < Math.min(10, nScenarios); sCheck++) {
                    if (cfArrays[sCheck] && cfArrays[sCheck][yCheck] > 0) {
                        yearHasData[yCheck] = 1;
                        break;
                    }
                }
            }

            var refHR = REFERENCE_HEAT_RATES[fuel] || 10.0;
            var plantHR = p.heat_rate_mmbtu_mwh || refHR;
            var efficiencyRatio = plantHR > 0 ? Math.min(refHR / plantHR, 1.5) : 1.0;

            var capMW = (p.capacity_mw || 0) * (p.equity_share || 1.0);
            var action = p._action;
            var yearOnline = p._year_online;
            var baseCO2 = p.co2_rate_t_mwh || DEFAULT_CO2_RATES[fuel] || 0.37;

            // Per-plant tracking for detail
            var genArr = [];
            var emisArr = [];
            var cfArr = [];
            for (var si = 0; si < nScenarios; si++) {
                genArr[si] = new Float64Array(nYears);
                emisArr[si] = new Float64Array(nYears);
                cfArr[si] = new Float64Array(nYears);
            }

            // Build per-year status
            var statusPerYear = [];
            for (var yi = 0; yi < nYears; yi++) {
                var y = years[yi];
                if (action === 'retire' && yearOnline && y >= yearOnline) {
                    statusPerYear[yi] = 'retired';
                } else if (action === 'ccs_retrofit' && yearOnline && y >= yearOnline) {
                    statusPerYear[yi] = 'ccs_retrofit';
                } else if (action === 'add_plant' && yearOnline && y < yearOnline) {
                    statusPerYear[yi] = 'not_yet_built';
                } else {
                    statusPerYear[yi] = p.status || 'operating';
                }
            }

            // Pre-compute CCS capture per year
            var capturePerYear = new Float64Array(nYears);
            // Use per-plant derate if set, otherwise fall back to global panel value
            var plantDeratePct = (p._ccs_derate_pct != null && p._ccs_derate_pct > 0) ? p._ccs_derate_pct : globalDeratePct;
            var derateFactor = 1.0 - plantDeratePct / 100.0; // e.g. 0.86 for 14% derate
            // Use per-plant capture if set, otherwise global
            var plantCaptureFrac = (p._ccs_target_rate > 0) ? p._ccs_target_rate : (globalCapturePct / 100.0);
            if (action === 'ccs_retrofit' && yearOnline) {
                for (var yi2 = 0; yi2 < nYears; yi2++) {
                    capturePerYear[yi2] = ccsRampFraction(years[yi2], yearOnline, plantCaptureFrac);
                }
            }

            // Dispatch across all scenarios × years
            for (var si2 = 0; si2 < nScenarios; si2++) {
                var scenarioCF = cfArrays[si2];
                var scenarioMargin = marginArrays[si2];

                for (var yi3 = 0; yi3 < nYears; yi3++) {
                    var baseCf = scenarioCF[yi3] || 0;
                    var margin = scenarioMargin[yi3] || 0;

                    // Fallback to static CF when sweep has no data for this year
                    if (!yearHasData[yi3]) {
                        baseCf = fallbackCf;
                        margin = 1; // Assume positive margin for baseline year
                    }

                    // Efficiency adjustment
                    var adjustedCf = Math.min(baseCf * efficiencyRatio, 0.95);

                    // Economic retirement (skip for fallback years)
                    if (yearHasData[yi3] && margin < 0) adjustedCf = 0;

                    // Year-aware masks
                    if (action === 'retire' && yearOnline && years[yi3] >= yearOnline) {
                        adjustedCf = 0;
                    }
                    if (action === 'add_plant' && yearOnline && years[yi3] < yearOnline) {
                        adjustedCf = 0;
                    }
                    // Plants marked retired in source data: run through 2024 (Mystic retired June 2024)
                    // then zero for 2025+
                    if (p.status === 'retired' && !action && years[yi3] > 2024) {
                        adjustedCf = 0;
                    }

                    // Generation
                    var genMwh = capMW * adjustedCf * 8760;

                    // CCS: runs baseload at user-set CF (to maximize 45Q), derate for parasitic load,
                    // capture rate reduces CO₂
                    var effectiveCO2 = baseCO2;
                    var reportFuel = fuel;
                    if (action === 'ccs_retrofit' && yearOnline) {
                        var capture = capturePerYear[yi3];
                        if (capture > 0) {
                            // CCS plants run baseload at the user-set CF, not market-driven
                            genMwh = capMW * globalCcsCfCap * 8760;
                            // Derate reduces generation (parasitic load)
                            genMwh = genMwh * derateFactor;
                            // Capture reduces CO₂ emissions
                            effectiveCO2 = baseCO2 * (1.0 - capture);
                            reportFuel = 'ccs_ccgt';
                        }
                    } else if (p.status === 'ccs_retrofit' && statusPerYear[yi3] === 'ccs_retrofit') {
                        // Plant already has CCS in base fleet config — only apply if status matches
                        var staticCapture = p.ccs_capture_rate || (globalCapturePct / 100.0);
                        genMwh = capMW * globalCcsCfCap * 8760;
                        genMwh = genMwh * derateFactor;
                        effectiveCO2 = baseCO2 * (1.0 - staticCapture);
                        reportFuel = 'ccs_ccgt';
                    }

                    var emisMt = genMwh * effectiveCO2 / 1e6;

                    // Derive effective CF from actual generation
                    var effectiveCf = (capMW > 0) ? genMwh / (capMW * 8760) : 0;

                    fleetEmissions[si2][yi3] += emisMt;
                    fleetGeneration[si2][yi3] += genMwh; // Accumulate total gen for intensity
                    genArr[si2][yi3] = genMwh;
                    emisArr[si2][yi3] = emisMt;
                    cfArr[si2][yi3] = effectiveCf;

                    // Accumulate by fuel
                    if (genByFuel[reportFuel]) {
                        genByFuel[reportFuel][si2][yi3] += genMwh / 1e6; // TWh
                    }
                    if (emisByFuel[reportFuel]) {
                        emisByFuel[reportFuel][si2][yi3] += emisMt;
                    }
                }
            }

            plantResults.push({
                plant: p,
                genArr: genArr,
                emisArr: emisArr,
                cfArr: cfArr,
                statusPerYear: statusPerYear
            });
        });

        // ── Build envelope (P10/P50/P90) ──
        var envelope = {};
        for (var yi4 = 0; yi4 < nYears; yi4++) {
            var col = [];
            for (var si3 = 0; si3 < nScenarios; si3++) {
                col.push(fleetEmissions[si3][yi4]);
            }
            envelope[String(years[yi4])] = {
                p10: Math.round(percentile(col, 10) * 10000) / 10000,
                p50: Math.round(percentile(col, 50) * 10000) / 10000,
                p90: Math.round(percentile(col, 90) * 10000) / 10000,
                mean: Math.round(col.reduce(function (a, b) { return a + b; }, 0) / col.length * 10000) / 10000
            };
        }

        // ── Build intensity envelope (kgCO2/MWh) = emissions / total generation ──
        var intensityEnvelope = {};
        for (var yi4b = 0; yi4b < nYears; yi4b++) {
            var intCol = [];
            for (var si3b = 0; si3b < nScenarios; si3b++) {
                var totalGenMwh = fleetGeneration[si3b][yi4b];
                var totalEmisMt = fleetEmissions[si3b][yi4b];
                // Mt CO2 → kg CO2: multiply by 1e9. Then divide by MWh.
                var intensityKg = totalGenMwh > 0 ? (totalEmisMt * 1e9) / totalGenMwh : 0;
                intCol.push(intensityKg);
            }
            intensityEnvelope[String(years[yi4b])] = {
                p10: Math.round(percentile(intCol, 10) * 100) / 100,
                p50: Math.round(percentile(intCol, 50) * 100) / 100,
                p90: Math.round(percentile(intCol, 90) * 100) / 100,
                mean: Math.round(intCol.reduce(function (a, b) { return a + b; }, 0) / intCol.length * 100) / 100
            };
        }

        // ── Build plant detail at P50 scenario ──
        var plantDetail = {};
        for (var yi5 = 0; yi5 < nYears; yi5++) {
            var col2 = [];
            for (var si4 = 0; si4 < nScenarios; si4++) {
                col2.push(fleetEmissions[si4][yi5]);
            }
            var medianVal = percentile(col2, 50);
            var bestSi = 0;
            var bestDiff = Infinity;
            for (var si5 = 0; si5 < nScenarios; si5++) {
                var d = Math.abs(fleetEmissions[si5][yi5] - medianVal);
                if (d < bestDiff) { bestDiff = d; bestSi = si5; }
            }

            var yearPlants = [];
            plantResults.forEach(function (pr) {
                yearPlants.push({
                    orispl: pr.plant.orispl,
                    name: pr.plant.name,
                    iso: pr.plant.iso,
                    fuel_type: pr.plant.fuel_type,
                    capacity_mw: Math.round((pr.plant.capacity_mw || 0) * (pr.plant.equity_share || 1) * 10) / 10,
                    status: pr.statusPerYear[yi5],
                    gen_twh: Math.round(pr.genArr[bestSi][yi5] / 1e6 * 100) / 100,
                    emissions_mt: Math.round(pr.emisArr[bestSi][yi5] * 10000) / 10000
                });
            });
            plantDetail[String(years[yi5])] = yearPlants;
        }

        // ── Build generation_by_fuel and emissions_by_fuel at P50 ──
        var generationByFuel = {};
        var emissionsByFuel = {};
        for (var yi6 = 0; yi6 < nYears; yi6++) {
            var yr = String(years[yi6]);

            // Find P50 scenario index for this year
            var col3 = [];
            for (var si6 = 0; si6 < nScenarios; si6++) {
                col3.push(fleetEmissions[si6][yi6]);
            }
            var medVal = percentile(col3, 50);
            var p50Si = 0;
            var p50Diff = Infinity;
            for (var si7 = 0; si7 < nScenarios; si7++) {
                var dd = Math.abs(fleetEmissions[si7][yi6] - medVal);
                if (dd < p50Diff) { p50Diff = dd; p50Si = si7; }
            }

            var genObj = {};
            var emisObj = {};
            ['coal_steam', 'gas_ccgt', 'gas_ct', 'oil_ct', 'gas_oil_ct', 'nuclear', 'geothermal', 'wind', 'solar', 'hydro', 'ccs_ccgt', 'battery', 'ldes'].forEach(function (f) {
                if (genByFuel[f]) {
                    genObj[f] = Math.round(genByFuel[f][p50Si][yi6] * 100) / 100;
                }
                if (emisByFuel[f]) {
                    emisObj[f] = Math.round(emisByFuel[f][p50Si][yi6] * 100) / 100;
                }
            });
            generationByFuel[yr] = genObj;
            emissionsByFuel[yr] = emisObj;
        }

        return {
            envelope: envelope,
            intensity_envelope: intensityEnvelope,
            plant_detail: plantDetail,
            generation_by_fuel: generationByFuel,
            emissions_by_fuel: emissionsByFuel,
            fleet_summary: {
                total_plants: activePlants.length,
                total_capacity_mw: Math.round(
                    activePlants.reduce(function (s, p) {
                        return s + (p.capacity_mw || 0) * (p.equity_share || 1);
                    }, 0) * 10
                ) / 10,
                n_scenarios: nScenarios,
                years: years.slice()
            }
        };
    }

    // ── Public API ──
    return {
        REFERENCE_HEAT_RATES: REFERENCE_HEAT_RATES,
        EMISSION_FACTORS: EMISSION_FACTORS,
        DEFAULT_CO2_RATES: DEFAULT_CO2_RATES,
        ccsRampFraction: ccsRampFraction,
        applyScenario: applyScenario,
        computeFleetDispatch: computeFleetDispatch,
        percentile: percentile
    };

})();
