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
    var lastComputedResults = null; // Cache most recent recalculation results

    var MAX_SCENARIOS = 5;
    var SCENARIO_STORAGE_KEY = 'market-sim-scenarios';
    var SCENARIO_COLORS = ['#2372B9', '#F47B27', '#6BA543', '#9C27B0', '#E91E63'];

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
        derate_pct: 14,       // 0-30%, default 14%
        capture_rate_pct: 90, // 50-99%, default 90%
        cf_pct: 85            // 20-95%, default 85% — max CF for CCS-retrofitted plants
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
        if (!fleetPlants.length && !addedPlants.length) {
            els.fleetList.innerHTML = '<div style="text-align:center;color:#6b7280;padding:24px;">No fleet data loaded</div>';
            return;
        }

        var allPlants = fleetPlants.concat(addedPlants);

        // Group by category first, then by ISO within each category
        var byCategoryISO = {};
        var categoryOrder = ['fossil', 'nuclear', 'renewable', 'storage', 'other'];
        allPlants.forEach(function (p) {
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
                    nuclear: '#6366F1',
                    renewable: '#22C55E',
                    fossil: '#6B7280',
                    storage: '#06B6D4'
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
                var currentAction = p._action || 'operating';
                var isFossil = FOSSIL_FUELS.has(p.fuel_type);
                var isCcsEligible = p.ccs_eligible === true;

                html += '<div class="sb-plant-row" data-plant-idx="' + idx + '">';
                html += '<div class="sb-plant-name" title="' + (p.full_name || p.name || '') + ' (' + fuelLabel + ')">';
                html += (p.name || 'Plant ' + idx);
                html += '<br><span class="sb-plant-cap">' + capStr + ' · ' + fuelLabel + equityStr + '</span>';
                html += '</div>';

                if (isFossil) {
                    // Fossil: editable capacity
                    html += '<div><input type="number" class="sb-cap-input" data-idx="' + idx + '" value="' + Math.round(p.capacity_mw || 0) + '" min="0" step="10" title="Capacity (MW)"></div>';

                    // Status dropdown
                    var modClass = (currentAction && currentAction !== 'operating' && currentAction !== p.status) ? ' modified' : '';
                    html += '<div><select class="sb-status-select' + modClass + '" data-idx="' + idx + '">';
                    html += '<option value="operating"' + (currentAction === 'operating' || !currentAction ? ' selected' : '') + '>Operating</option>';
                    html += '<option value="retire"' + (currentAction === 'retire' ? ' selected' : '') + '>Retired</option>';
                    if (isCcsEligible) {
                        html += '<option value="ccs_retrofit"' + (currentAction === 'ccs_retrofit' ? ' selected' : '') + '>CCS Retrofit</option>';
                    }
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
                    html += '<div><select class="sb-status-select' + ((currentAction && currentAction !== 'operating') ? ' modified' : '') + '" data-idx="' + idx + '">';
                    html += '<option value="operating"' + (currentAction === 'operating' || !currentAction ? ' selected' : '') + '>Operating</option>';
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

        // Bind ISO group collapse + auto-collapse non-fossil categories
        els.fleetList.querySelectorAll('.sb-iso-header').forEach(function (header) {
            header.addEventListener('click', function () {
                var key = this.dataset.iso;
                var body = els.fleetList.querySelector('[data-iso-body="' + key + '"]');
                if (body) {
                    var collapsed = body.classList.toggle('collapsed');
                    this.classList.toggle('collapsed', collapsed);
                }
            });

            // Auto-collapse non-fossil categories so only fossil is expanded by default
            var key = header.dataset.iso;
            var cat = key.split('|')[0];
            if (cat !== 'fossil') {
                var body = els.fleetList.querySelector('[data-iso-body="' + key + '"]');
                if (body) {
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
        p._action = newStatus === 'operating' ? null : newStatus;
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
        if (newStatus === 'operating') {
            p._ccs_target_rate = 0;
            p._ccs_derate_pct = 0;
            p._uprate_mw = 0;
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
        nuclear: 92, geothermal: 85, solar: 22, wind: 30,
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

        // Reset form
        document.getElementById('apName').value = '';
        document.getElementById('apCapacity').value = '1200';
        document.getElementById('apHeatRate').value = '6.5';
        document.getElementById('apYearOnline').value = '2028';

        renderFleetList();
        setStatus('Added "' + name + '" (' + fuelType + ') in ' + iso + ' — click Recalculate to update charts');
    }

    // ── Recalculate ──
    function recalculate() {
        if (!sweepData) {
            setStatus('Error: Sweep data not loaded');
            return;
        }

        var allPlants = fleetPlants.concat(addedPlants);
        if (!allPlants.length) {
            setStatus('Error: No plants in fleet');
            return;
        }

        setStatus('Calculating...');
        if (els.recalcBtn) {
            els.recalcBtn.disabled = true;
            els.recalcBtn.textContent = 'Calculating...';
        }

        requestAnimationFrame(function () {
            setTimeout(function () {
                try {
                    var t0 = performance.now();
                    var result = FleetDispatchEngine.computeFleetDispatch(allPlants, sweepData, {
                        ccs_derate_pct: ccsParams.derate_pct,
                        ccs_capture_rate_pct: ccsParams.capture_rate_pct,
                        ccs_cf_pct: ccsParams.cf_pct
                    });
                    var elapsed = Math.round(performance.now() - t0);

                    // Cache the results and current params for save
                    lastComputedResults = {
                        envelope: result.envelope,
                        plant_detail: result.plant_detail,
                        generation_by_fuel: result.generation_by_fuel,
                        emissions_by_fuel: result.emissions_by_fuel,
                        fleet_summary: result.fleet_summary
                    };

                    var scenarioName = (els.nameInput && els.nameInput.value.trim()) || 'Custom';

                    var scenarioData = {
                        description: scenarioName,
                        color: '#2372B9',
                        envelope: result.envelope,
                        plant_detail: result.plant_detail,
                        generation_by_fuel: result.generation_by_fuel,
                        emissions_by_fuel: result.emissions_by_fuel,
                        fleet_summary: result.fleet_summary
                    };

                    if (window.FLEET_SCENARIOS_API) {
                        window.FLEET_SCENARIOS_API.setCustomScenario(scenarioName, scenarioData);
                        setStatus('Done in ' + elapsed + 'ms — custom scenario updated');
                    } else {
                        console.warn('FLEET_SCENARIOS_API not available');
                        setStatus('Computed in ' + elapsed + 'ms but chart API unavailable');
                    }
                    updateSaveButtonState();
                } catch (err) {
                    console.error('Recalculation failed:', err);
                    setStatus('Error: ' + err.message);
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
        els.saveBtn.disabled = !name || atMax;
        if (atMax) {
            els.saveBtn.title = 'Maximum ' + MAX_SCENARIOS + ' scenarios reached';
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
                return { _idx: p._idx, orispl: p.orispl, _action: p._action, _year_online: p._year_online, _ccs_target_rate: p._ccs_target_rate, capacity_mw: p.capacity_mw };
            }),
            addedPlants: JSON.parse(JSON.stringify(addedPlants))
        };

        // If we haven't computed results yet, run recalculate first
        if (!lastComputedResults && sweepData) {
            recalculate();
        }

        var scenario = {
            id: generateId(),
            name: name,
            description: '',
            params: params,
            results: lastComputedResults ? JSON.parse(JSON.stringify(lastComputedResults)) : null,
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
    function syncVisibleScenarios() {
        if (!window.FLEET_SCENARIOS_API) return;
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
        getFleetPlants: function () { return fleetPlants.concat(addedPlants); }
    };

})();
