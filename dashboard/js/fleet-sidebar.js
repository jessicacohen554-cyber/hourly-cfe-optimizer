// ============================================================================
// FLEET SIDEBAR — Controller for fleet configuration sidebar on fleet-scenarios
// ============================================================================
// Loads base fleet + sweep dispatch data, renders editable plant list,
// runs client-side recalculation via FleetDispatchEngine, and injects
// results back into the chart system via FLEET_SCENARIOS_API.
// ============================================================================

var FleetSidebar = (function () {
    'use strict';

    // ── State ──
    var baseFleet = [];        // Original fleet from constellation_scenarios.json
    var fleetPlants = [];      // Working copy with modifications
    var addedPlants = [];      // User-added plants
    var sweepData = null;      // From sweep_dispatch_data.json
    var scenarioConfig = null; // Full constellation_scenarios.json
    var savedScenarios = [];   // Array of SavedScenario objects (max 5)
    var isOpen = false;
    var nextAddId = 90000;     // ID counter for added plants
    var openGroups = new Set(); // Track which accordion groups are open across re-renders
    var lastComputedResults = null; // Cache most recent recalculation results

    var MAX_SCENARIOS = 8;
    var SCENARIO_STORAGE_KEY = 'market-sim-scenarios-v2';
    var SCENARIO_COLORS = ['#2372B9', '#F47B27', '#6BA543', '#651dda', '#E91E63', '#007FA4', '#CADB2E', '#14B8A6'];

    // ── Fuel labels ──
    var FUEL_LABELS = {
        gas_ccgt: 'CCGT', gas_ct: 'CT', oil_ct: 'Oil', gas_oil_ct: 'Gas/Oil',
        coal_steam: 'Coal', ccs_ccgt: 'CCS-CCGT',
        nuclear: 'Nuclear', geothermal: 'Geothermal', wind: 'Wind',
        solar: 'Solar', hydro: 'Hydro', battery: 'Battery',
        battery_4hr: 'Battery 4hr', battery_8hr: 'Battery 8hr',
        ldes: 'LDES (Iron-Air)', pumped_storage: 'Pumped Storage'
    };

    var FOSSIL_FUELS = new Set(['gas_ccgt', 'gas_ct', 'oil_ct', 'gas_oil_ct']);

    // ── DOM refs ──
    var els = {};

    // ── Init ──
    function init() {
        cacheElements();
        bindEvents();
        loadData();
    }

    // ── CCS global parameters ──
    var ccsParams = {
        derate_pct: 14,       // 0-30%, default 13%
        capture_rate_pct: 90, // 50-99%, default 95%
        cf_pct: 85            // 20-95%, default 80% — max CF for CCS-retrofitted plants
    };

    function cacheElements() {
        els.sidebar = document.getElementById('fleetSidebar');
        els.backdrop = document.getElementById('sidebarBackdrop');
        els.toggle = document.getElementById('sidebarToggle');
        els.close = document.getElementById('sidebarClose');
        els.fleetList = document.getElementById('sidebarFleetList');
        els.recalcBtn = document.getElementById('recalcBtn');
        els.resetBtn = document.getElementById('resetBaselineBtn');
        els.saveBtn = document.getElementById('saveScenarioBtn');
        els.addBtn = document.getElementById('addPlantBtn');
        els.nameInput = document.getElementById('scenarioNameInput');
        els.status = document.getElementById('recalcStatus');
        els.savedList = document.getElementById('savedScenariosList');

        // CCS panel elements
        els.ccsDerateSlider = document.getElementById('ccsDerateSlider');
        els.ccsDerateValue = document.getElementById('ccsDerateValue');
        els.ccsCaptureSlider = document.getElementById('ccsCaptureSlider');
        els.ccsCaptureValue = document.getElementById('ccsCaptureValue');
        els.ccsCfSlider = document.getElementById('ccsCfSlider');
        els.ccsCfValue = document.getElementById('ccsCfValue');
        els.ccsApplyAllBtn = document.getElementById('ccsApplyAllBtn');
    }

    function bindEvents() {
        if (els.toggle) els.toggle.addEventListener('click', open);
        if (els.close) els.close.addEventListener('click', close);
        if (els.backdrop) els.backdrop.addEventListener('click', close);
        if (els.recalcBtn) els.recalcBtn.addEventListener('click', recalculate);
        if (els.resetBtn) els.resetBtn.addEventListener('click', function () { resetFleet(); });
        if (els.saveBtn) els.saveBtn.addEventListener('click', saveScenario);
        if (els.addBtn) els.addBtn.addEventListener('click', addPlant);
        if (els.nameInput) {
            els.nameInput.addEventListener('input', updateSaveButtonState);
        }

        // CCS parameter sliders
        if (els.ccsDerateSlider) {
            els.ccsDerateSlider.addEventListener('input', function () {
                ccsParams.derate_pct = parseInt(this.value);
                if (els.ccsDerateValue) els.ccsDerateValue.textContent = ccsParams.derate_pct + '%';
            });
        }
        if (els.ccsCaptureSlider) {
            els.ccsCaptureSlider.addEventListener('input', function () {
                ccsParams.capture_rate_pct = parseInt(this.value);
                if (els.ccsCaptureValue) els.ccsCaptureValue.textContent = ccsParams.capture_rate_pct + '%';
            });
        }
        if (els.ccsCfSlider) {
            els.ccsCfSlider.addEventListener('input', function () {
                ccsParams.cf_pct = parseInt(this.value);
                if (els.ccsCfValue) els.ccsCfValue.textContent = ccsParams.cf_pct + '%';
            });
        }
        if (els.ccsApplyAllBtn) {
            els.ccsApplyAllBtn.addEventListener('click', applyCcsToAll);
        }

        // Keyboard: Escape to close
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape' && isOpen) close();
        });
    }

    // ── Apply CCS params to all CCS-eligible plants ──
    function applyCcsToAll() {
        var count = 0;
        fleetPlants.forEach(function (p) {
            if (p.ccs_eligible && FOSSIL_FUELS.has(p.fuel_type)) {
                p._action = 'ccs_retrofit';
                if (!p._year_online) p._year_online = 2030;
                p._ccs_target_rate = ccsParams.capture_rate_pct / 100.0;
                p._ccs_derate_pct = ccsParams.derate_pct;
                count++;
            }
        });
        renderFleetList();
        setStatus('Applied CCS (' + ccsParams.derate_pct + '% derate, ' +
            ccsParams.capture_rate_pct + '% capture) to ' + count + ' plants — click Recalculate');
    }

    // ── Data Loading ──
    function loadData() {
        var fleetLoaded = false, sweepLoaded = false;

        // 1. Fleet scenarios config (base fleet + scenario definitions)
        fetch('data/constellation_scenarios.json')
            .then(function (r) { if (r.ok) return r.json(); throw new Error('No config'); })
            .then(function (data) {
                scenarioConfig = data;
                baseFleet = (data.base_fleet || []).map(function (p, i) {
                    return Object.assign({}, p, {
                        _idx: i,
                        _action: null,
                        _year_online: null,
                        _ccs_target_rate: 0
                    });
                });
                fleetPlants = JSON.parse(JSON.stringify(baseFleet));
                fleetLoaded = true;
                if (sweepLoaded) onAllLoaded();
                renderFleetList();
            })
            .catch(function (err) {
                console.warn('Fleet config not available:', err);
                fleetLoaded = true;
                if (sweepLoaded) onAllLoaded();
            });

        // 2. Sweep dispatch data
        fetch('data/sweep_dispatch_data.json')
            .then(function (r) { if (r.ok) return r.json(); throw new Error('No sweep data'); })
            .then(function (data) {
                sweepData = data;
                sweepLoaded = true;
                if (fleetLoaded) onAllLoaded();
            })
            .catch(function (err) {
                console.warn('Sweep dispatch data not available:', err);
                sweepLoaded = true;
                if (fleetLoaded) onAllLoaded();
            });

        // Load saved scenarios from localStorage
        try {
            var saved = localStorage.getItem(SCENARIO_STORAGE_KEY);
            if (saved) {
                var parsed = JSON.parse(saved);
                // Handle migration from old object format
                if (Array.isArray(parsed)) {
                    savedScenarios = parsed;
                } else {
                    savedScenarios = [];
                }
            }
        } catch (e) { savedScenarios = []; }
    }

    function loadFleetFallback(callback) {
        fetch('data/constellation_scenarios.json')
            .then(function (r) { if (r.ok) return r.json(); throw new Error('No fleet'); })
            .then(function (plants) {
                var fossilFuels = { 'Gas': 'gas_ccgt', 'Oil': 'oil_ct', 'Gas/Oil': 'gas_oil_ct' };
                baseFleet = plants
                    .filter(function (p) { return fossilFuels[p.fuel_type]; })
                    .map(function (p, i) {
                        var fuel = fossilFuels[p.fuel_type];
                        var hr = FleetDispatchEngine.REFERENCE_HEAT_RATES[fuel] || 10.0;
                        var ef = FleetDispatchEngine.EMISSION_FACTORS[fuel] || 0.05306;
                        return {
                            orispl: parseInt(p.id) || i,
                            name: p.name,
                            iso: p.iso,
                            capacity_mw: p.capacity_mw || 0,
                            fuel_type: fuel,
                            heat_rate_mmbtu_mwh: hr,
                            co2_rate_t_mwh: Math.round(hr * ef * 100000) / 100000,
                            equity_share: p.equity_pct || 1.0,
                            status: (p.status || 'Operating').toLowerCase(),
                            ccs_capture_rate: 0,
                            ccs_heat_rate_penalty: 1.0,
                            _idx: i,
                            _action: null,
                            _year_online: null,
                            _ccs_target_rate: 0
                        };
                    });
                fleetPlants = JSON.parse(JSON.stringify(baseFleet));
                renderFleetList();
                if (callback) callback();
            })
            .catch(function () {
                console.warn('No fleet data available');
                if (callback) callback();
            });
    }

    function onAllLoaded() {
        renderSavedScenarios();
        updateSaveButtonState();
        // Push any previously saved visible scenarios to charts
        setTimeout(syncVisibleScenarios, 200);

        // Count by category
        var nuclearCount = 0, renewableCount = 0, fossilCount = 0, storageCount = 0;
        baseFleet.forEach(function (p) {
            var cat = p.plant_category || (FOSSIL_FUELS.has(p.fuel_type) ? 'fossil' : 'other');
            if (cat === 'nuclear') nuclearCount++;
            else if (cat === 'renewable') renewableCount++;
            else if (cat === 'fossil') fossilCount++;
            else if (cat === 'storage') storageCount++;
        });

        var statusMsg = 'Ready — ' + baseFleet.length + ' plants (' +
            nuclearCount + ' nuclear, ' + renewableCount + ' renewable, ' +
            fossilCount + ' fossil, ' + storageCount + ' storage)';
        if (sweepData) {
            statusMsg += ', ' + (sweepData.n_scenarios || 0) + ' sweep scenarios';
        }
        setStatus(statusMsg);

        // Compute baseline via the same dispatch engine used for custom scenarios.
        // This ensures deltas between baseline and user modifications are accurate
        // (apples-to-apples comparison using the same methodology).
        if (sweepData) {
            computeAndSetBaseline();
        }

        // ── Migration: recompute results for saved scenarios that have null results ──
        // (fixes scenarios saved before the async race condition fix)
        migrateNullResultScenarios();
    }

    function migrateNullResultScenarios() {
        if (!window.FLEET_SCENARIOS_API) {
            // API not ready yet — retry once after a delay
            setTimeout(migrateNullResultScenarios, 500);
            return;
        }
        var precomputed = window.FLEET_SCENARIOS_API.getData();
        if (!precomputed || !precomputed.scenarios || !precomputed.scenarios.baseline) return;

        var baseline = precomputed.scenarios.baseline;
        var migrated = 0;

        savedScenarios.forEach(function (s) {
            if (s.results) return; // Already has results
            if (!s.params) return; // No params to recompute from

            // Reconstruct fleet state from saved params
            var tempFleet = JSON.parse(JSON.stringify(baseFleet));
            (s.params.fleetMods || []).forEach(function (mod) {
                var p = tempFleet[mod._idx];
                if (p && p.orispl === mod.orispl) {
                    p._action = mod._action;
                    p._year_online = mod._year_online;
                    p._ccs_target_rate = mod._ccs_target_rate;
                    p._ccs_derate_pct = mod._ccs_derate_pct || 0;
                    p._uprate_mw = mod._uprate_mw || 0;
                    p._custom_cf = mod._custom_cf || 0;
                    if (mod.capacity_mw != null) p.capacity_mw = mod.capacity_mw;
                }
            });
            var tempAdded = JSON.parse(JSON.stringify(s.params.addedPlants || []));
            var allPlants = tempFleet.concat(tempAdded);

            try {
                if (sweepData && sweepData.years) {
                    s.results = FleetDispatchEngine.computeFleetDispatch(allPlants, sweepData, {
                        ccs_derate_pct: ccsParams.derate_pct,
                        ccs_capture_rate_pct: ccsParams.capture_rate_pct,
                        ccs_cf_pct: ccsParams.cf_pct
                    });
                } else {
                    s.results = computeScenarioResults(allPlants, tempAdded, baseline, ccsParams);
                }
                migrated++;
                console.log('[fleet-sidebar] Migrated scenario "' + s.name + '" — recomputed results from params');
            } catch (err) {
                console.warn('[fleet-sidebar] Failed to migrate scenario "' + s.name + '":', err);
            }
        });

        if (migrated > 0) {
            persistScenarios();
            renderSavedScenarios();
            syncVisibleScenarios();
            console.log('[fleet-sidebar] Migrated ' + migrated + ' scenarios with null results');
        }
    }

    function computeAndSetBaseline() {
        // Use unmodified base fleet (no _action flags) — this IS the market trajectory
        var unmodifiedFleet = baseFleet.map(function (p) {
            return Object.assign({}, p, {
                _action: null,
                _year_online: null,
                _ccs_target_rate: 0,
                _uprate_mw: 0
            });
        });

        try {
            var result = FleetDispatchEngine.computeFleetDispatch(unmodifiedFleet, sweepData, {
                ccs_derate_pct: 0,
                ccs_capture_rate_pct: 0,
                ccs_cf_pct: 85
            });

            // Push to chart system as the BASELINE (replacing precomputed JSON baseline)
            if (window.FLEET_SCENARIOS_API && window.FLEET_SCENARIOS_API.setComputedBaseline) {
                window.FLEET_SCENARIOS_API.setComputedBaseline({
                    envelope: result.envelope,
                    intensity_envelope: result.intensity_envelope,
                    plant_detail: result.plant_detail,
                    generation_by_fuel: result.generation_by_fuel,
                    emissions_by_fuel: result.emissions_by_fuel,
                    fleet_summary: result.fleet_summary
                });
                console.log('Baseline computed via dispatch engine — unified with custom scenario path');
            }
        } catch (err) {
            console.warn('Baseline auto-compute failed, falling back to precomputed JSON:', err);
        }
    }

    // ── Sidebar Open/Close ──
    function open() {
        isOpen = true;
        if (els.sidebar) els.sidebar.classList.add('open');
        if (els.backdrop) els.backdrop.classList.add('open');
        document.body.style.overflow = 'hidden';
    }

    function close() {
        isOpen = false;
        if (els.sidebar) els.sidebar.classList.remove('open');
        if (els.backdrop) els.backdrop.classList.remove('open');
        document.body.style.overflow = '';
    }

    // ── Render Fleet List ──
    function renderFleetList() {
        if (!els.fleetList) return;

        // Snapshot open groups before DOM rebuild
        var prevOpen = new Set();
        els.fleetList.querySelectorAll('.sb-iso-body').forEach(function(body) {
            if (!body.classList.contains('collapsed')) {
                prevOpen.add(body.dataset.isoBody);
            }
        });
        if (prevOpen.size > 0) openGroups = prevOpen;

        // Capture dirty input values before DOM rebuild so they aren't lost
        els.fleetList.querySelectorAll('.sb-uprate-input').forEach(function (inp) {
            var p = findPlant(inp.dataset.idx);
            if (p) p._uprate_mw = Math.max(0, parseFloat(inp.value) || 0);
        });
        els.fleetList.querySelectorAll('.sb-year-input').forEach(function (inp) {
            var p = findPlant(inp.dataset.idx);
            if (p) p._year_online = parseInt(inp.value) || null;
        });
        els.fleetList.querySelectorAll('.sb-cap-input').forEach(function (inp) {
            var p = findPlant(inp.dataset.idx);
            if (p) p.capacity_mw = Math.max(0, parseFloat(inp.value) || 0);
        });

        if (!fleetPlants.length && !addedPlants.length) {
            els.fleetList.innerHTML = '<div style="text-align:center;color:#6b7280;padding:24px;">No fleet data loaded</div>';
            return;
        }

        var allPlants = fleetPlants.concat(addedPlants);

        // Filter out retired plants — they exist in data for 2023 baseline
        // emissions but shouldn't appear in the sidebar for user interaction
        var visiblePlants = allPlants.filter(function (p) {
            return p.status !== 'retired';
        });

        // Group by category first, then by ISO within each category
        var byCategoryISO = {};
        var categoryOrder = ['fossil', 'nuclear', 'renewable', 'storage', 'other'];
        visiblePlants.forEach(function (p) {
            var cat = p.plant_category || (FOSSIL_FUELS.has(p.fuel_type) ? 'fossil' : 'other');
            var iso = p.iso || 'Other';
            var key = cat + '|' + iso;
            if (!byCategoryISO[key]) byCategoryISO[key] = { category: cat, iso: iso, plants: [] };
            byCategoryISO[key].plants.push(p);
        });

        // Sort groups by category order, then ISO order
        var isoOrder = ['PJM', 'ERCOT', 'CAISO', 'NYISO', 'NEISO', 'MISO', 'SPP'];
        var sortedKeys = Object.keys(byCategoryISO).sort(function (a, b) {
            var ga = byCategoryISO[a], gb = byCategoryISO[b];
            var catA = categoryOrder.indexOf(ga.category), catB = categoryOrder.indexOf(gb.category);
            if (catA < 0) catA = 99;
            if (catB < 0) catB = 99;
            if (catA !== catB) return catA - catB;
            var isoA = isoOrder.indexOf(ga.iso), isoB = isoOrder.indexOf(gb.iso);
            if (isoA < 0) isoA = 99;
            if (isoB < 0) isoB = 99;
            return isoA - isoB;
        });

        var html = '';
        var lastCategory = null;

        sortedKeys.forEach(function (key) {
            var group = byCategoryISO[key];
            var plants = group.plants;
            var totalMW = plants.reduce(function (s, p) { return s + (p.capacity_mw || 0); }, 0);

            // Category header
            if (group.category !== lastCategory) {
                lastCategory = group.category;
                var catLabels = {
                    nuclear: 'Nuclear Fleet',
                    renewable: 'Renewables & Geothermal',
                    fossil: 'Fossil Fleet',
                    storage: 'Energy Storage'
                };
                var catColors = {
                    nuclear: '#2372B9',
                    renewable: '#6BA543',
                    fossil: '#7E8083',
                    storage: '#651dda'
                };
                html += '<div class="sb-category-header" style="background:' + (catColors[group.category] || '#888') + ';color:#fff;padding:8px 12px;font-weight:700;font-size:0.85rem;margin-top:8px;border-radius:6px 6px 0 0;">';
                html += (catLabels[group.category] || group.category);
                html += '</div>';
            }

            html += '<div class="sb-iso-group">';
            html += '<div class="sb-iso-header" data-iso="' + key + '">';
            html += '<span>' + group.iso + ' (' + plants.length + ' plants, ' + Math.round(totalMW).toLocaleString() + ' MW)</span>';
            html += '<span class="sb-chevron">▼</span>';
            html += '</div>';
            html += '<div class="sb-iso-body" data-iso-body="' + key + '">';

            plants.forEach(function (p) {
                var isAdded = p._isAdded;
                var idx = isAdded ? 'add_' + p._addId : p._idx;
                var fuelLabel = FUEL_LABELS[p._original_fuel_type] || FUEL_LABELS[p.fuel_type] || p.fuel_type;
                var capStr = Math.round(p.capacity_mw || 0) + ' MW';
                var equityStr = p.equity_share < 1 ? ' (' + Math.round(p.equity_share * 100) + '% equity)' : '';
                var currentAction = p._action || 'default_market';
                var isFossil = FOSSIL_FUELS.has(p.fuel_type);

                html += '<div class="sb-plant-row" data-plant-idx="' + idx + '">';
                html += '<div class="sb-plant-name" title="' + (p.full_name || p.name || '') + ' (' + fuelLabel + ')">';
                html += (p.name || 'Plant ' + idx);
                html += '<br><span class="sb-plant-cap">' + capStr + ' · ' + fuelLabel + equityStr + '</span>';
                html += '</div>';

                if (isFossil) {
                    // Fossil: editable capacity
                    html += '<div><input type="number" class="sb-cap-input" data-idx="' + idx + '" value="' + Math.round(p.capacity_mw || 0) + '" min="0" step="10" title="Capacity (MW)"></div>';

                    // Status dropdown
                    var modClass = (currentAction && currentAction !== 'default_market' && currentAction !== p.status) ? ' modified' : '';
                    html += '<div><select class="sb-status-select' + modClass + '" data-idx="' + idx + '">';
                    html += '<option value="default_market"' + (currentAction === 'default_market' ? ' selected' : '') + '>Default Market</option>';
                    html += '<option value="operating"' + (currentAction === 'operating_override' ? ' selected' : '') + '>Operating (forced)</option>';
                    html += '<option value="retire"' + (currentAction === 'retire' ? ' selected' : '') + '>Retired</option>';
                    html += '<option value="ccs_retrofit"' + (currentAction === 'ccs_retrofit' ? ' selected' : '') + '>CCS Retrofit</option>';
                    html += '</select></div>';

                    // Year input (for retire/CCS)
                    var showYear = (currentAction === 'retire' || currentAction === 'ccs_retrofit' || currentAction === 'add_plant');
                    var yearVal = p._year_online || 2030;
                    html += '<div>';
                    if (showYear) {
                        html += '<input type="number" class="sb-year-input" data-idx="' + idx + '" value="' + yearVal + '" min="2023" max="2050" title="Year">';
                    }
                    if (isAdded) {
                        html += '<button class="sb-btn-danger sb-remove-added" data-add-id="' + p._addId + '" title="Remove">✕</button>';
                    }
                    html += '</div>';
                } else {
                    var isNuclear = p.fuel_type === 'nuclear';

                    // Non-fossil: read-only capacity (plus uprate MW if applicable)
                    if (isNuclear && currentAction === 'uprate') {
                        var uprateMW = p._uprate_mw || 0;
                        html += '<div style="display:flex;flex-direction:column;gap:2px;">';
                        html += '<span class="sb-cap-readonly" style="font-size:0.78rem;color:#6b7280;">' + capStr + '</span>';
                        html += '<input type="number" class="sb-uprate-input" data-idx="' + idx + '" value="' + Math.round(uprateMW) + '" min="0" max="2000" step="10" placeholder="+MW" title="Additional uprate capacity (MW)" style="width:72px;font-size:0.75rem;padding:2px 4px;border:1px solid #c7d2e8;border-radius:4px;">';
                        html += '</div>';
                    } else {
                        html += '<div><span class="sb-cap-readonly" style="font-size:0.78rem;color:#6b7280;">' + capStr + '</span></div>';
                    }

                    // Status: Operating / Retired / Uprate (nuclear only)
                    html += '<div><select class="sb-status-select' + ((currentAction && currentAction !== 'default_market') ? ' modified' : '') + '" data-idx="' + idx + '">';
                    html += '<option value="default_market"' + (currentAction === 'default_market' ? ' selected' : '') + '>Default Market</option>';
                    html += '<option value="operating"' + (currentAction === 'operating_override' ? ' selected' : '') + '>Operating (forced)</option>';
                    html += '<option value="retire"' + (currentAction === 'retire' ? ' selected' : '') + '>Retired</option>';
                    if (isNuclear) {
                        html += '<option value="uprate"' + (currentAction === 'uprate' ? ' selected' : '') + '>Uprate</option>';
                    }
                    html += '</select></div>';

                    // Year input for retirement or uprate
                    var showYearNF = (currentAction === 'retire' || currentAction === 'uprate');
                    var yearValNF = p._year_online || 2030;
                    html += '<div>';
                    if (showYearNF) {
                        html += '<input type="number" class="sb-year-input" data-idx="' + idx + '" value="' + yearValNF + '" min="2023" max="2050" title="Year">';
                    }
                    html += '</div>';
                }

                html += '</div>';
            });

            html += '</div></div>';
        });

        els.fleetList.innerHTML = html;

        // Bind ISO group collapse + restore open/collapsed state
        els.fleetList.querySelectorAll('.sb-iso-header').forEach(function (header) {
            header.addEventListener('click', function () {
                var key = this.dataset.iso;
                var body = els.fleetList.querySelector('[data-iso-body="' + key + '"]');
                if (body) {
                    var collapsed = body.classList.toggle('collapsed');
                    this.classList.toggle('collapsed', collapsed);
                    if (collapsed) {
                        openGroups.delete(key);
                    } else {
                        openGroups.add(key);
                    }
                }
            });

            // Restore state: if openGroups has entries, use them; otherwise default all collapsed
            var key = header.dataset.iso;
            var body = els.fleetList.querySelector('[data-iso-body="' + key + '"]');
            if (body) {
                if (openGroups.size > 0) {
                    var shouldCollapse = !openGroups.has(key);
                    body.classList.toggle('collapsed', shouldCollapse);
                    header.classList.toggle('collapsed', shouldCollapse);
                } else {
                    body.classList.add('collapsed');
                    header.classList.add('collapsed');
                }
            }
        });

        // Bind status selects
        els.fleetList.querySelectorAll('.sb-status-select').forEach(function (sel) {
            sel.addEventListener('change', function () {
                onStatusChange(this.dataset.idx, this.value);
            });
        });

        // Bind year inputs
        els.fleetList.querySelectorAll('.sb-year-input').forEach(function (inp) {
            inp.addEventListener('change', function () {
                onYearChange(this.dataset.idx, parseInt(this.value));
            });
        });

        // Bind capacity inputs
        els.fleetList.querySelectorAll('.sb-cap-input').forEach(function (inp) {
            inp.addEventListener('change', function () {
                onCapacityChange(this.dataset.idx, parseFloat(this.value));
            });
        });

        // Bind uprate MW inputs
        els.fleetList.querySelectorAll('.sb-uprate-input').forEach(function (inp) {
            inp.addEventListener('change', function () {
                var p = findPlant(this.dataset.idx);
                if (p) p._uprate_mw = Math.max(0, parseFloat(this.value) || 0);
            });
        });

        // Bind remove buttons for added plants
        els.fleetList.querySelectorAll('.sb-remove-added').forEach(function (btn) {
            btn.addEventListener('click', function () {
                var addId = parseInt(this.dataset.addId);
                addedPlants = addedPlants.filter(function (p) { return p._addId !== addId; });
                renderFleetList();
            });
        });
    }

    // ── Plant Modification Handlers ──
    function findPlant(idx) {
        if (String(idx).startsWith('add_')) {
            var addId = parseInt(String(idx).replace('add_', ''));
            return addedPlants.find(function (p) { return p._addId === addId; });
        }
        return fleetPlants[parseInt(idx)];
    }

    function onStatusChange(idx, newStatus) {
        var p = findPlant(idx);
        if (!p) return;
        if (newStatus === 'default_market') {
            // Default Market: clear all action flags — plant follows sweep economics
            p._action = null;
            p._year_online = null;
            p._ccs_target_rate = 0;
            p._ccs_derate_pct = 0;
            p._uprate_mw = 0;
        } else if (newStatus === 'operating') {
            // Operating (forced): override economic retirement
            p._action = 'operating_override';
            p._ccs_target_rate = 0;
            p._ccs_derate_pct = 0;
            p._uprate_mw = 0;
        } else {
            p._action = newStatus;
            if (newStatus === 'retire' || newStatus === 'ccs_retrofit' || newStatus === 'uprate') {
                if (!p._year_online) p._year_online = 2030;
                if (newStatus === 'ccs_retrofit') {
                    // Use global CCS panel params
                    p._ccs_target_rate = ccsParams.capture_rate_pct / 100.0;
                    p._ccs_derate_pct = ccsParams.derate_pct;
                }
                if (newStatus === 'uprate') {
                    if (!p._uprate_mw) p._uprate_mw = 200; // Default uprate MW
                }
            }
        }
        renderFleetList();
    }

    function onYearChange(idx, year) {
        var p = findPlant(idx);
        if (p) p._year_online = year;
    }

    function onCapacityChange(idx, cap) {
        var p = findPlant(idx);
        if (p) p.capacity_mw = cap;
    }

    // ── Fuel type → category mapping ──
    var FOSSIL_FUEL_TYPES = new Set(['gas_ccgt', 'gas_ct', 'oil_ct', 'gas_oil_ct', 'coal_steam']);
    var STORAGE_FUEL_TYPES = new Set(['battery_4hr', 'battery_8hr', 'ldes']);
    var DEFAULT_CF = {
        nuclear: 90, geothermal: 74, solar: 22, wind: 16,
        battery_4hr: 0, battery_8hr: 0, ldes: 0
    };

    // Regional CFs (%) — mirrors dispatch engine REGIONAL_CF
    var REGIONAL_CF_PCT = {
        solar: { CAISO: 27, ERCOT: 24, PJM: 18, NYISO: 16, NEISO: 16, MISO: 20, SPP: 23 },
        wind:  { CAISO: 26, ERCOT: 35, PJM: 28, NYISO: 28, NEISO: 30, MISO: 34, SPP: 38 }
    };

    // Show/hide form fields based on selected fuel type
    function onFuelTypeChanged() {
        var sel = document.getElementById('apFuelType');
        if (!sel) return;
        var ft = sel.value;
        var isFossil = FOSSIL_FUEL_TYPES.has(ft);
        var isStorage = STORAGE_FUEL_TYPES.has(ft);
        var isClean = !isFossil && !isStorage;

        var fossilFields = document.getElementById('apFossilFields');
        var cleanFields = document.getElementById('apCleanFields');
        if (fossilFields) fossilFields.style.display = isFossil ? '' : 'none';
        if (cleanFields) cleanFields.style.display = isClean ? '' : 'none';

        // Update default CF for clean types — use regional value if available
        if (isClean) {
            var cfInput = document.getElementById('apCF');
            var isoSel = document.getElementById('apISO');
            var iso = isoSel ? isoSel.value : '';
            var regionalVal = REGIONAL_CF_PCT[ft] && REGIONAL_CF_PCT[ft][iso];
            if (cfInput) cfInput.value = regionalVal || DEFAULT_CF[ft] || 30;
        }

        // Update placeholder and default capacity
        var nameInput = document.getElementById('apName');
        var capInput = document.getElementById('apCapacity');
        var labels = {
            gas_ccgt: ['New CCGT', 1200], gas_ct: ['New CT', 400], oil_ct: ['New Oil CT', 200],
            gas_oil_ct: ['New Dual Fuel', 400], nuclear: ['New Nuclear', 1100],
            geothermal: ['New Geothermal', 50], solar: ['New Solar Farm', 500],
            wind: ['New Wind Farm', 300], battery_4hr: ['New Battery 4hr', 200],
            battery_8hr: ['New Battery 8hr', 200], ldes: ['New LDES', 100]
        };
        if (nameInput && labels[ft]) nameInput.placeholder = labels[ft][0];
        if (capInput && labels[ft]) capInput.value = labels[ft][1];
    }
    window._apFuelTypeChanged = onFuelTypeChanged;

    // ── Add Plant ──
    function addPlant() {
        var name = document.getElementById('apName').value.trim();
        var iso = document.getElementById('apISO').value;
        var capacity = parseFloat(document.getElementById('apCapacity').value);
        var fuelType = document.getElementById('apFuelType').value;
        var yearOnline = parseInt(document.getElementById('apYearOnline').value);

        if (!name) { alert('Please enter a plant name'); return; }
        if (!capacity || capacity <= 0) { alert('Please enter a valid capacity'); return; }
        if (!yearOnline || yearOnline < 2023 || yearOnline > 2050) { alert('Year must be 2023-2050'); return; }

        var isFossil = FOSSIL_FUEL_TYPES.has(fuelType);
        var isStorage = STORAGE_FUEL_TYPES.has(fuelType);

        // Validate fossil-specific fields
        var heatRate = 0;
        var co2Rate = 0;
        if (isFossil) {
            heatRate = parseFloat(document.getElementById('apHeatRate').value);
            if (!heatRate || heatRate <= 0) { alert('Please enter a valid heat rate'); return; }
            var ef = FleetDispatchEngine.EMISSION_FACTORS[fuelType] || 0.05306;
            co2Rate = Math.round(heatRate * ef * 100000) / 100000;
        }

        // Get custom CF for clean types
        var customCF = 0;
        if (!isFossil && !isStorage) {
            customCF = parseFloat(document.getElementById('apCF').value) / 100.0;
            if (customCF <= 0 || customCF > 1) { alert('Capacity factor must be 1-100%'); return; }
        }

        // Normalize storage fuel types to dispatch engine names
        var dispatchFuel = fuelType;
        if (fuelType === 'battery_4hr' || fuelType === 'battery_8hr') dispatchFuel = 'battery';
        // LDES stays as 'ldes'

        var category;
        if (isFossil) category = 'fossil';
        else if (isStorage) category = 'storage';
        else if (fuelType === 'nuclear') category = 'nuclear';
        else if (fuelType === 'geothermal') category = 'nuclear'; // Groups with clean firm
        else category = 'renewable';

        var newPlant = {
            orispl: nextAddId,
            name: name,
            iso: iso,
            capacity_mw: capacity,
            fuel_type: dispatchFuel,
            _original_fuel_type: fuelType, // preserve 4hr vs 8hr distinction
            plant_category: category,
            heat_rate_mmbtu_mwh: heatRate,
            co2_rate_t_mwh: co2Rate,
            equity_share: 1.0,
            status: 'operating',
            ccs_capture_rate: 0,
            ccs_heat_rate_penalty: 1.0,
            ccs_eligible: isFossil && fuelType === 'gas_ccgt',
            _action: 'add_plant',
            _year_online: yearOnline,
            _ccs_target_rate: 0,
            _custom_cf: customCF,
            _isAdded: true,
            _addId: nextAddId
        };
        nextAddId++;
        addedPlants.push(newPlant);

        console.log('[fleet-sidebar] addPlant:', newPlant.name,
            '| fuel:', newPlant.fuel_type,
            '| capacity:', newPlant.capacity_mw, 'MW',
            '| cf:', newPlant._custom_cf,
            '| co2_rate:', newPlant.co2_rate_t_mwh,
            '| year:', newPlant._year_online,
            '| addedPlants count:', addedPlants.length);

        // Reset form
        document.getElementById('apName').value = '';
        document.getElementById('apCapacity').value = '1200';
        document.getElementById('apHeatRate').value = '6.5';
        document.getElementById('apYearOnline').value = '2028';

        renderFleetList();
        setStatus('Added "' + name + '" (' + fuelType + ') in ' + iso + ' — click Recalculate to update charts');
    }

    // ── Core computation (synchronous) ──
    // Takes fleet + added plants + baseline data → returns results object
    function computeScenarioResults(allPlantsList, addedPlantsList, baseline, ccsP) {
        var years = Object.keys(baseline.envelope).sort();

        // Build orispl → plant AND name → plant lookup for modified plants
        var modifiedPlants = {};
        var modifiedByName = {};
        var restartByName = {};
        allPlantsList.forEach(function (p) {
            if (p._action && p._action !== 'default_market') {
                modifiedPlants[p.orispl] = p;
                if (p.name) modifiedByName[p.name] = p;
            }
            if (p.year_built && p.year_built > 2024 && !FOSSIL_FUELS.has(p.fuel_type)) {
                if (p.name) restartByName[p.name] = p;
            }
        });

        var newPlants = addedPlantsList.filter(function (p) {
            return p._action === 'add_plant';
        });

        var customEnvelope = {};
        var customIntensity = {};
        var customPlantDetail = {};
        var customGenByFuel = {};
        var customEmisByFuel = {};

        years.forEach(function (yr) {
            var yearNum = parseInt(yr);
            var basePlants = baseline.plant_detail[yr] || [];
            var baseEnv = baseline.envelope[yr] || { p10: 0, p50: 0, p90: 0 };
            var baseGen = baseline.generation_by_fuel[yr] || {};
            var baseEmis = baseline.emissions_by_fuel[yr] || {};

            var genByFuel = {};
            Object.keys(baseGen).forEach(function (f) { genByFuel[f] = baseGen[f]; });
            var emisByFuel = {};
            Object.keys(baseEmis).forEach(function (f) { emisByFuel[f] = baseEmis[f]; });

            var emisDelta = 0;
            var genDelta = 0;
            var yearPlants = [];
            var newCleanGen = {};

            basePlants.forEach(function (bp) {
                var mod = modifiedPlants[bp.orispl] || modifiedByName[bp.name];
                if (!mod) {
                    yearPlants.push({
                        orispl: bp.orispl, name: bp.name, iso: bp.iso,
                        fuel_type: bp.fuel_type, capacity_mw: bp.capacity_mw,
                        status: bp.status, gen_twh: bp.gen_twh, emissions_mt: bp.emissions_mt
                    });
                    var restartPlant = restartByName[bp.name];
                    if (restartPlant && bp.gen_twh > 0 && yearNum >= (restartPlant.year_built || 9999)) {
                        var rFuel = bp.fuel_type || restartPlant.fuel_type;
                        if (!FOSSIL_FUELS.has(rFuel)) {
                            newCleanGen[rFuel] = (newCleanGen[rFuel] || 0) + bp.gen_twh;
                        }
                    }
                    return;
                }

                var action = mod._action;
                var yearOnline = mod._year_online || 2030;
                var fuel = bp.fuel_type || mod.fuel_type;
                var capMW = (mod.capacity_mw || bp.capacity_mw || 0) * (mod.equity_share || 1.0);
                var co2Rate = (mod.co2_rate_t_mwh != null) ? mod.co2_rate_t_mwh : ((bp.co2_rate_t_mwh != null) ? bp.co2_rate_t_mwh : 0.37);
                var customGenTwh = bp.gen_twh;
                var customEmisMt = bp.emissions_mt;
                var customStatus = bp.status;
                var customFuel = fuel;

                if (action === 'retire' && yearNum >= yearOnline) {
                    customGenTwh = 0;
                    customEmisMt = 0;
                    customStatus = 'retired';
                } else if (action === 'ccs_retrofit' && yearNum >= yearOnline) {
                    var cfFrac = (ccsP.cf_pct || 85) / 100.0;
                    var derateFrac = (ccsP.derate_pct || 14) / 100.0;
                    var captureFrac = (mod._ccs_target_rate > 0) ?
                        mod._ccs_target_rate : (ccsP.capture_rate_pct || 90) / 100.0;
                    var grossMwh = capMW * cfFrac * 8760;
                    var netMwh = grossMwh * (1.0 - derateFrac);
                    var emisTons = grossMwh * co2Rate * (1.0 - captureFrac);
                    customGenTwh = netMwh / 1e6;
                    customEmisMt = emisTons / 1e6;
                    customStatus = 'ccs_retrofit';
                    customFuel = 'ccs_ccgt';
                } else if (action === 'uprate' && yearNum >= yearOnline) {
                    var uprateMW = mod._uprate_mw || 0;
                    var origMW = bp.capacity_mw || capMW;
                    if (origMW > 0) {
                        var scale = (origMW + uprateMW) / origMW;
                        customGenTwh = bp.gen_twh * scale;
                        customEmisMt = bp.emissions_mt * scale;
                    }
                } else if (action === 'operating_override') {
                    if (bp.gen_twh <= 0 && capMW > 0) {
                        var forcedCf = mod._custom_cf || 0.90;
                        customGenTwh = capMW * forcedCf * 8760 / 1e6;
                        customEmisMt = FOSSIL_FUELS.has(fuel) ? (capMW * forcedCf * 8760 * co2Rate / 1e6) : 0;
                    }
                }

                var isCleanFuel = !FOSSIL_FUELS.has(customFuel);
                if (action === 'operating_override' && isCleanFuel) {
                    newCleanGen[customFuel] = (newCleanGen[customFuel] || 0) + customGenTwh;
                } else if (action === 'uprate' && isCleanFuel && yearNum >= yearOnline) {
                    var uprateIncrement = customGenTwh - (bp.gen_twh || 0);
                    if (uprateIncrement > 0) {
                        newCleanGen[customFuel] = (newCleanGen[customFuel] || 0) + uprateIncrement;
                    }
                } else if (action === 'ccs_retrofit' && yearNum >= yearOnline) {
                    newCleanGen['ccs_ccgt'] = (newCleanGen['ccs_ccgt'] || 0) + customGenTwh;
                }

                var dEmis = customEmisMt - (bp.emissions_mt || 0);
                var dGen = customGenTwh - (bp.gen_twh || 0);
                emisDelta += dEmis;
                genDelta += dGen;

                var oldFuel = bp.fuel_type || 'gas_ccgt';
                if (customFuel !== oldFuel) {
                    emisByFuel[oldFuel] = (emisByFuel[oldFuel] || 0) - (bp.emissions_mt || 0);
                    emisByFuel[customFuel] = (emisByFuel[customFuel] || 0) + customEmisMt;
                    genByFuel[oldFuel] = (genByFuel[oldFuel] || 0) - (bp.gen_twh || 0);
                    genByFuel[customFuel] = (genByFuel[customFuel] || 0) + customGenTwh;
                } else {
                    emisByFuel[oldFuel] = (emisByFuel[oldFuel] || 0) + dEmis;
                    genByFuel[oldFuel] = (genByFuel[oldFuel] || 0) + dGen;
                }

                yearPlants.push({
                    orispl: bp.orispl, name: bp.name, iso: bp.iso || mod.iso,
                    fuel_type: customFuel, capacity_mw: bp.capacity_mw,
                    status: customStatus,
                    gen_twh: Math.round(customGenTwh * 100) / 100,
                    emissions_mt: Math.round(customEmisMt * 10000) / 10000
                });
            });

            // Process new (added) plants
            newPlants.forEach(function (np) {
                var yearOnline = np._year_online || 2028;
                if (yearNum < yearOnline) return;
                var fuel = np.fuel_type || 'gas_ccgt';
                var capMW = (np.capacity_mw || 0) * (np.equity_share || 1.0);
                var co2Rate = (np.co2_rate_t_mwh != null) ? np.co2_rate_t_mwh : 0.37;
                var cf = (np._custom_cf && np._custom_cf > 0) ? np._custom_cf : 0.57;
                var grossMwh = capMW * cf * 8760;
                var emisTons = grossMwh * co2Rate;
                var genTwh = grossMwh / 1e6;
                var emisMt = emisTons / 1e6;

                emisDelta += emisMt;
                genDelta += genTwh;
                emisByFuel[fuel] = (emisByFuel[fuel] || 0) + emisMt;
                genByFuel[fuel] = (genByFuel[fuel] || 0) + genTwh;

                if (!FOSSIL_FUELS.has(fuel)) {
                    newCleanGen[fuel] = (newCleanGen[fuel] || 0) + genTwh;
                }

                yearPlants.push({
                    orispl: np.orispl, name: np.name, iso: np.iso,
                    fuel_type: fuel, capacity_mw: np.capacity_mw,
                    status: 'operating',
                    gen_twh: Math.round(genTwh * 100) / 100,
                    emissions_mt: Math.round(emisMt * 10000) / 10000
                });
            });

            customEnvelope[yr] = {
                p10: Math.round((baseEnv.p10 + emisDelta) * 10000) / 10000,
                p50: Math.round((baseEnv.p50 + emisDelta) * 10000) / 10000,
                p90: Math.round((baseEnv.p90 + emisDelta) * 10000) / 10000
            };

            var totalGenTwh = 0;
            var totalEmisMt = 0;
            Object.keys(genByFuel).forEach(function (f) { if (f[0] !== '_') totalGenTwh += genByFuel[f] || 0; });
            Object.keys(emisByFuel).forEach(function (f) { if (f[0] !== '_') totalEmisMt += emisByFuel[f] || 0; });
            var intensityKg = totalGenTwh > 0 ? (totalEmisMt / totalGenTwh) * 1e3 : 0;
            customIntensity[yr] = {
                p10: Math.round(intensityKg * 100) / 100,
                p50: Math.round(intensityKg * 100) / 100,
                p90: Math.round(intensityKg * 100) / 100
            };

            genByFuel._new_clean_gen = newCleanGen;
            customPlantDetail[yr] = yearPlants;
            customGenByFuel[yr] = genByFuel;
            customEmisByFuel[yr] = emisByFuel;
        });

        return {
            envelope: customEnvelope,
            intensity_envelope: customIntensity,
            plant_detail: customPlantDetail,
            generation_by_fuel: customGenByFuel,
            emissions_by_fuel: customEmisByFuel
        };
    }

    // ── Recalculate ──
    // Full dispatch engine approach: runs all plants (modified + added) through the
    // same FleetDispatchEngine.computeFleetDispatch() used for the baseline. This
    // gives proper P10/P50/P90 envelopes from the parametric sweep dispatch data,
    // with heat-rate-adjusted capacity factors for new fossil plants.
    function recalculate() {
        if (!window.FLEET_SCENARIOS_API) {
            setStatus('Error: Chart API not available');
            return;
        }
        if (!sweepData || !sweepData.years) {
            setStatus('Error: Sweep dispatch data not loaded');
            return;
        }

        var allPlants = fleetPlants.concat(addedPlants);
        if (!allPlants.length) {
            setStatus('Error: No plants in fleet');
            return;
        }

        console.log('[fleet-sidebar] recalculate starting (dispatch engine):',
            '| fleetPlants:', fleetPlants.length,
            '| addedPlants:', addedPlants.length,
            '| allPlants:', allPlants.length,
            '| sweepScenarios:', sweepData.n_scenarios,
            '| sweepYears:', sweepData.years.length);

        setStatus('Calculating...');
        if (els.recalcBtn) {
            els.recalcBtn.disabled = true;
            els.recalcBtn.textContent = 'Calculating...';
        }

        requestAnimationFrame(function () {
            setTimeout(function () {
                try {
                    var t0 = performance.now();

                    // Run the full dispatch engine — same path as baseline, but with
                    // user modifications and added plants included in the fleet.
                    lastComputedResults = FleetDispatchEngine.computeFleetDispatch(
                        allPlants, sweepData, {
                            ccs_derate_pct: ccsParams.derate_pct,
                            ccs_capture_rate_pct: ccsParams.capture_rate_pct,
                            ccs_cf_pct: ccsParams.cf_pct
                        }
                    );

                    var elapsed = Math.round(performance.now() - t0);

                    // Debug log
                    var dbgGen2030 = lastComputedResults.generation_by_fuel['2030'];
                    var dbgInt2030 = lastComputedResults.intensity_envelope['2030'];
                    console.log('[fleet-sidebar] recalculate complete (dispatch engine):',
                        '| addedPlants:', addedPlants.length,
                        '| elapsed:', elapsed + 'ms',
                        '| 2030 gen_by_fuel:', dbgGen2030 ? JSON.stringify(dbgGen2030) : 'MISSING',
                        '| 2030 intensity:', dbgInt2030 ? dbgInt2030.p50 + ' kg/MWh' : 'MISSING',
                        '| 2030 envelope:', lastComputedResults.envelope['2030'] ? JSON.stringify(lastComputedResults.envelope['2030']) : 'MISSING');

                    setStatus('Computed in ' + elapsed + 'ms — name & save to display on charts');
                    updateSaveButtonState();
                    // Auto-focus name input to encourage saving
                    if (els.nameInput && !els.nameInput.value.trim()) {
                        els.nameInput.focus();
                    }
                } catch (err) {
                    console.error('Recalculation failed:', err, err.stack);
                    setStatus('Error: ' + err.message + ' (see console for details)');
                }

                if (els.recalcBtn) {
                    els.recalcBtn.disabled = false;
                    els.recalcBtn.textContent = 'Recalculate';
                }
            }, 50);
        });
    }

    // ── UUID generator ──
    function generateId() {
        if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
            var r = Math.random() * 16 | 0;
            return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
        });
    }

    // ── Get next available color ──
    function getNextColor() {
        var usedColors = savedScenarios.map(function (s) { return s.color; });
        for (var i = 0; i < SCENARIO_COLORS.length; i++) {
            if (usedColors.indexOf(SCENARIO_COLORS[i]) === -1) return SCENARIO_COLORS[i];
        }
        return SCENARIO_COLORS[0];
    }

    // ── Save button state ──
    function updateSaveButtonState() {
        if (!els.saveBtn) return;
        var name = (els.nameInput && els.nameInput.value.trim()) || '';
        var atMax = savedScenarios.length >= MAX_SCENARIOS;
        var noResults = !lastComputedResults;
        els.saveBtn.disabled = !name || atMax || noResults;
        if (atMax) {
            els.saveBtn.title = 'Maximum ' + MAX_SCENARIOS + ' scenarios reached';
        } else if (noResults) {
            els.saveBtn.title = 'Click Recalculate first to compute results';
        } else if (!name) {
            els.saveBtn.title = 'Enter a name to save';
        } else {
            els.saveBtn.title = '';
        }
    }

    // ── Persist to localStorage ──
    function persistScenarios() {
        try {
            localStorage.setItem(SCENARIO_STORAGE_KEY, JSON.stringify(savedScenarios));
        } catch (e) { console.warn('localStorage save failed:', e); }
    }

    // ── Save/Load Scenarios ──
    function saveScenario() {
        var name = (els.nameInput && els.nameInput.value.trim()) || '';
        if (!name) { alert('Please enter a scenario name'); return; }
        if (savedScenarios.length >= MAX_SCENARIOS) {
            alert('Maximum ' + MAX_SCENARIOS + ' scenarios. Delete one to save a new one.');
            return;
        }

        // Build params from current fleet state
        var params = {
            fleetMods: fleetPlants.filter(function (p) { return p._action; }).map(function (p) {
                return { _idx: p._idx, orispl: p.orispl, _action: p._action, _year_online: p._year_online, _ccs_target_rate: p._ccs_target_rate, _ccs_derate_pct: p._ccs_derate_pct, _uprate_mw: p._uprate_mw, _custom_cf: p._custom_cf, capacity_mw: p.capacity_mw };
            }),
            addedPlants: JSON.parse(JSON.stringify(addedPlants))
        };

        // Require recalculation before saving — recalculate is async so we can't
        // auto-run it here and expect results to be ready synchronously
        if (!lastComputedResults) {
            alert('Please click "Recalculate" first to compute results before saving.');
            return;
        }

        var scenario = {
            id: generateId(),
            name: name,
            description: '',
            params: params,
            results: JSON.parse(JSON.stringify(lastComputedResults)),
            color: getNextColor(),
            isVisible: true,
            isBaseline: savedScenarios.length === 0, // First scenario is baseline
            createdAt: new Date().toISOString()
        };

        savedScenarios.push(scenario);
        persistScenarios();
        renderSavedScenarios();
        updateSaveButtonState();
        syncVisibleScenarios();
        if (els.nameInput) els.nameInput.value = '';
        setStatus('Saved "' + name + '" (' + savedScenarios.length + '/' + MAX_SCENARIOS + ')');
    }

    function loadScenario(id) {
        var saved = savedScenarios.find(function (s) { return s.id === id; });
        if (!saved || !saved.params) return;

        fleetPlants = JSON.parse(JSON.stringify(baseFleet));

        (saved.params.fleetMods || []).forEach(function (mod) {
            var p = fleetPlants[mod._idx];
            if (p && p.orispl === mod.orispl) {
                p._action = mod._action;
                p._year_online = mod._year_online;
                p._ccs_target_rate = mod._ccs_target_rate;
                p._ccs_derate_pct = mod._ccs_derate_pct || 0;
                p._uprate_mw = mod._uprate_mw || 0;
                p._custom_cf = mod._custom_cf || 0;
                if (mod.capacity_mw != null) p.capacity_mw = mod.capacity_mw;
            }
        });

        addedPlants = JSON.parse(JSON.stringify(saved.params.addedPlants || []));
        lastComputedResults = null; // Clear — user must recalculate after tweaking

        renderFleetList();
        setStatus('Loaded "' + saved.name + '" — tweak & Recalculate, or save as new');
    }

    function deleteScenario(id) {
        var idx = savedScenarios.findIndex(function (s) { return s.id === id; });
        if (idx === -1) return;
        var name = savedScenarios[idx].name;
        savedScenarios.splice(idx, 1);
        persistScenarios();
        renderSavedScenarios();
        updateSaveButtonState();
        syncVisibleScenarios();
        setStatus('Deleted "' + name + '"');
    }

    function toggleVisibility(id) {
        var s = savedScenarios.find(function (s) { return s.id === id; });
        if (!s) return;
        s.isVisible = !s.isVisible;
        persistScenarios();
        renderSavedScenarios();
        syncVisibleScenarios();
    }

    function setBaseline(id) {
        savedScenarios.forEach(function (s) {
            s.isBaseline = (s.id === id);
        });
        persistScenarios();
        renderSavedScenarios();
        syncVisibleScenarios();
    }

    function updateScenarioName(id, newName) {
        var s = savedScenarios.find(function (s) { return s.id === id; });
        if (!s) return;
        s.name = newName.trim() || s.name;
        persistScenarios();
        renderSavedScenarios();
        syncVisibleScenarios();
    }

    // ── Push visible scenarios to chart system ──
    var _syncRetries = 0;
    function syncVisibleScenarios() {
        if (!window.FLEET_SCENARIOS_API) {
            // API not ready yet — retry with backoff (up to ~5 seconds total)
            if (_syncRetries < 8) {
                _syncRetries++;
                setTimeout(syncVisibleScenarios, 250 * _syncRetries);
            }
            return;
        }
        _syncRetries = 0;
        var visible = savedScenarios.filter(function (s) { return s.isVisible && s.results; });
        window.FLEET_SCENARIOS_API.setSavedScenarios(visible);
    }

    // ── Escape HTML for safe rendering ──
    function escapeHtml(str) {
        var div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    function renderSavedScenarios() {
        if (!els.savedList) return;

        // Update count
        var countEl = document.getElementById('saveCount');
        if (countEl) {
            countEl.textContent = savedScenarios.length + '/' + MAX_SCENARIOS + ' scenarios saved';
        }

        if (!savedScenarios.length) {
            els.savedList.innerHTML = '<div style="color:#9ca3af;font-size:0.82rem;padding:8px 0;">No saved scenarios yet</div>';
            return;
        }

        var html = '';
        savedScenarios.forEach(function (s) {
            var baselineClass = s.isBaseline ? ' is-baseline' : '';
            html += '<div class="sb-saved-item' + baselineClass + '" data-scenario-id="' + s.id + '">';

            // Color swatch
            html += '<span class="sb-scenario-swatch" style="background:' + s.color + '"></span>';

            // Name (click to load params)
            html += '<span class="sb-scenario-name" data-load-id="' + s.id + '" title="Click to load params">' + escapeHtml(s.name) + '</span>';

            // Baseline badge
            if (s.isBaseline) {
                html += '<span class="sb-baseline-badge">Baseline</span>';
            }

            // Action buttons
            html += '<div class="sb-saved-actions">';

            // Visibility toggle (eye)
            var eyeIcon = s.isVisible ? '&#128065;' : '&#128065;&#xFE0E;';
            var eyeClass = s.isVisible ? ' active' : '';
            html += '<button class="sb-icon-btn' + eyeClass + '" data-toggle-id="' + s.id + '" title="' + (s.isVisible ? 'Hide from chart' : 'Show on chart') + '">';
            html += s.isVisible ? '👁' : '👁‍🗨';
            html += '</button>';

            // Set baseline
            if (!s.isBaseline) {
                html += '<button class="sb-icon-btn sb-icon-baseline" data-baseline-id="' + s.id + '" title="Set as baseline reference">★</button>';
            }

            // Delete
            html += '<button class="sb-icon-btn sb-icon-delete" data-delete-id="' + s.id + '" title="Delete scenario">✕</button>';

            html += '</div></div>';
        });

        els.savedList.innerHTML = html;

        // Bind click-to-load (on name)
        els.savedList.querySelectorAll('[data-load-id]').forEach(function (el) {
            el.addEventListener('click', function () { loadScenario(this.dataset.loadId); });
            // Double-click to rename
            el.addEventListener('dblclick', function (e) {
                e.stopPropagation();
                var id = this.dataset.loadId;
                var s = savedScenarios.find(function (sc) { return sc.id === id; });
                if (!s) return;
                var nameEl = this;
                var input = document.createElement('input');
                input.type = 'text';
                input.className = 'sb-scenario-name-input';
                input.value = s.name;
                input.maxLength = 60;
                nameEl.replaceWith(input);
                input.focus();
                input.select();

                function finishRename() {
                    var newName = input.value.trim();
                    if (newName && newName !== s.name) {
                        updateScenarioName(id, newName);
                    } else {
                        renderSavedScenarios();
                    }
                }
                input.addEventListener('blur', finishRename);
                input.addEventListener('keydown', function (ke) {
                    if (ke.key === 'Enter') { ke.preventDefault(); input.blur(); }
                    if (ke.key === 'Escape') { input.value = s.name; input.blur(); }
                });
            });
        });

        // Bind visibility toggle
        els.savedList.querySelectorAll('[data-toggle-id]').forEach(function (btn) {
            btn.addEventListener('click', function (e) {
                e.stopPropagation();
                toggleVisibility(this.dataset.toggleId);
            });
        });

        // Bind set baseline
        els.savedList.querySelectorAll('[data-baseline-id]').forEach(function (btn) {
            btn.addEventListener('click', function (e) {
                e.stopPropagation();
                setBaseline(this.dataset.baselineId);
            });
        });

        // Bind delete
        els.savedList.querySelectorAll('[data-delete-id]').forEach(function (btn) {
            btn.addEventListener('click', function (e) {
                e.stopPropagation();
                var id = this.dataset.deleteId;
                var s = savedScenarios.find(function (sc) { return sc.id === id; });
                if (s && confirm('Delete "' + s.name + '"?')) {
                    deleteScenario(id);
                }
            });
        });
    }

    // ── Reset fleet to base ──
    function resetFleet() {
        fleetPlants = JSON.parse(JSON.stringify(baseFleet));
        addedPlants = [];
        lastComputedResults = null;
        renderFleetList();
        // Clear custom scenario from charts
        if (window.FLEET_SCENARIOS_API && window.FLEET_SCENARIOS_API.clearCustomScenario) {
            window.FLEET_SCENARIOS_API.clearCustomScenario();
        }
        setStatus('Fleet reset to baseline');
    }

    // ── Status helper ──
    function setStatus(msg) {
        if (els.status) els.status.textContent = msg;
    }

    // ── Public API ──
    return {
        init: init,
        open: open,
        close: close,
        recalculate: recalculate,
        resetFleet: resetFleet,
        getSavedScenarios: function () { return savedScenarios; },
        getVisibleScenarios: function () {
            return savedScenarios.filter(function (s) { return s.isVisible && s.results; });
        },
        getFleetPlants: function () { return fleetPlants.concat(addedPlants); },
        getSweepData: function () { return sweepData; },
        getCcsParams: function () { return Object.assign({}, ccsParams); }
    };

})();
