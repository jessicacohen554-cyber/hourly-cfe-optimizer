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
    var savedScenarios = {};   // localStorage
    var isOpen = false;
    var nextAddId = 90000;     // ID counter for added plants

    // ── DOM refs ──
    var els = {};

    // ── Init ──
    function init() {
        cacheElements();
        bindEvents();
        loadData();
    }

    function cacheElements() {
        els.sidebar = document.getElementById('fleetSidebar');
        els.backdrop = document.getElementById('sidebarBackdrop');
        els.toggle = document.getElementById('sidebarToggle');
        els.close = document.getElementById('sidebarClose');
        els.fleetList = document.getElementById('sidebarFleetList');
        els.recalcBtn = document.getElementById('recalcBtn');
        els.saveBtn = document.getElementById('saveScenarioBtn');
        els.addBtn = document.getElementById('addPlantBtn');
        els.nameInput = document.getElementById('scenarioNameInput');
        els.status = document.getElementById('recalcStatus');
        els.savedList = document.getElementById('savedScenariosList');
    }

    function bindEvents() {
        if (els.toggle) els.toggle.addEventListener('click', open);
        if (els.close) els.close.addEventListener('click', close);
        if (els.backdrop) els.backdrop.addEventListener('click', close);
        if (els.recalcBtn) els.recalcBtn.addEventListener('click', recalculate);
        if (els.saveBtn) els.saveBtn.addEventListener('click', saveScenario);
        if (els.addBtn) els.addBtn.addEventListener('click', addPlant);

        // Keyboard: Escape to close
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape' && isOpen) close();
        });
    }

    // ── Data Loading ──
    function loadData() {
        // Load both in parallel
        var fleetLoaded = false, sweepLoaded = false;

        // 1. Fleet scenarios config (base fleet + scenario definitions)
        fetch('/api/fleet-scenarios-config')
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
                // Try loading from constellation_fleet.json as fallback
                loadFleetFallback(function () {
                    fleetLoaded = true;
                    if (sweepLoaded) onAllLoaded();
                });
            });

        // 2. Sweep dispatch data
        fetch('/api/sweep-dispatch-data')
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
            var saved = localStorage.getItem('fleet_custom_scenarios');
            if (saved) savedScenarios = JSON.parse(saved);
        } catch (e) { savedScenarios = {}; }
    }

    function loadFleetFallback(callback) {
        fetch('/api/fleet-config')
            .then(function (r) { if (r.ok) return r.json(); throw new Error('No fleet'); })
            .then(function (plants) {
                // Convert from fleet-config format to dispatch format
                var fossilFuels = { 'Gas': 'gas_ccgt', 'Coal': 'coal_steam', 'Oil': 'oil_ct', 'Gas/Oil': 'gas_ct', 'Oil/Coal': 'oil_ct' };
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
        if (sweepData) {
            setStatus('Ready — ' + baseFleet.length + ' plants, ' +
                (sweepData.n_scenarios || 0) + ' scenarios');
        } else {
            setStatus('Sweep data unavailable — recalculation disabled');
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
        if (!fleetPlants.length && !addedPlants.length) {
            els.fleetList.innerHTML = '<div style="text-align:center;color:#6b7280;padding:24px;">No fleet data loaded</div>';
            return;
        }

        var allPlants = fleetPlants.concat(addedPlants);

        // Group by ISO
        var byISO = {};
        allPlants.forEach(function (p) {
            var iso = p.iso || 'Other';
            if (!byISO[iso]) byISO[iso] = [];
            byISO[iso].push(p);
        });

        var isoOrder = ['PJM', 'ERCOT', 'CAISO', 'NYISO', 'NEISO', 'MISO', 'SPP'];
        var sortedISOs = Object.keys(byISO).sort(function (a, b) {
            var ai = isoOrder.indexOf(a), bi = isoOrder.indexOf(b);
            if (ai >= 0 && bi >= 0) return ai - bi;
            if (ai >= 0) return -1;
            if (bi >= 0) return 1;
            return a.localeCompare(b);
        });

        var html = '';
        sortedISOs.forEach(function (iso) {
            var plants = byISO[iso];
            var totalMW = plants.reduce(function (s, p) { return s + (p.capacity_mw || 0); }, 0);

            html += '<div class="sb-iso-group">';
            html += '<div class="sb-iso-header" data-iso="' + iso + '">';
            html += '<span>' + iso + ' (' + plants.length + ' plants, ' + Math.round(totalMW).toLocaleString() + ' MW)</span>';
            html += '<span class="sb-chevron">▼</span>';
            html += '</div>';
            html += '<div class="sb-iso-body" data-iso-body="' + iso + '">';

            plants.forEach(function (p) {
                var isAdded = p._isAdded;
                var idx = isAdded ? 'add_' + p._addId : p._idx;
                var fuelLabel = { gas_ccgt: 'CCGT', gas_ct: 'CT', coal_steam: 'Coal', oil_ct: 'Oil' };
                var capStr = Math.round(p.capacity_mw || 0) + ' MW';
                var currentAction = p._action || 'operating';

                html += '<div class="sb-plant-row" data-plant-idx="' + idx + '">';
                html += '<div class="sb-plant-name" title="' + (p.name || '') + ' (' + (fuelLabel[p.fuel_type] || p.fuel_type) + ')">';
                html += (p.name || 'Plant ' + idx);
                html += '<br><span class="sb-plant-cap">' + capStr + ' · ' + (fuelLabel[p.fuel_type] || p.fuel_type) + '</span>';
                html += '</div>';

                // Capacity input (for editing)
                html += '<div><input type="number" class="sb-cap-input" data-idx="' + idx + '" value="' + Math.round(p.capacity_mw || 0) + '" min="0" step="10" title="Capacity (MW)"></div>';

                // Status dropdown
                var modClass = (currentAction && currentAction !== 'operating' && currentAction !== p.status) ? ' modified' : '';
                html += '<div><select class="sb-status-select' + modClass + '" data-idx="' + idx + '">';
                html += '<option value="operating"' + (currentAction === 'operating' || !currentAction ? ' selected' : '') + '>Operating</option>';
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

                html += '</div>';
            });

            html += '</div></div>';
        });

        els.fleetList.innerHTML = html;

        // Bind ISO group collapse
        els.fleetList.querySelectorAll('.sb-iso-header').forEach(function (header) {
            header.addEventListener('click', function () {
                var iso = this.dataset.iso;
                var body = els.fleetList.querySelector('[data-iso-body="' + iso + '"]');
                if (body) {
                    var collapsed = body.classList.toggle('collapsed');
                    this.classList.toggle('collapsed', collapsed);
                }
            });
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
        if (newStatus === 'retire' || newStatus === 'ccs_retrofit') {
            if (!p._year_online) p._year_online = 2030;
            if (newStatus === 'ccs_retrofit') p._ccs_target_rate = 0.95;
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

    // ── Add Plant ──
    function addPlant() {
        var name = document.getElementById('apName').value.trim();
        var iso = document.getElementById('apISO').value;
        var capacity = parseFloat(document.getElementById('apCapacity').value);
        var heatRate = parseFloat(document.getElementById('apHeatRate').value);
        var fuelType = document.getElementById('apFuelType').value;
        var yearOnline = parseInt(document.getElementById('apYearOnline').value);

        if (!name) { alert('Please enter a plant name'); return; }
        if (!capacity || capacity <= 0) { alert('Please enter a valid capacity'); return; }
        if (!heatRate || heatRate <= 0) { alert('Please enter a valid heat rate'); return; }
        if (!yearOnline || yearOnline < 2023 || yearOnline > 2050) { alert('Year must be 2023-2050'); return; }

        var ef = FleetDispatchEngine.EMISSION_FACTORS[fuelType] || 0.05306;
        var co2Rate = Math.round(heatRate * ef * 100000) / 100000;

        var newPlant = {
            orispl: nextAddId,
            name: name,
            iso: iso,
            capacity_mw: capacity,
            fuel_type: fuelType,
            heat_rate_mmbtu_mwh: heatRate,
            co2_rate_t_mwh: co2Rate,
            equity_share: 1.0,
            status: 'operating',
            ccs_capture_rate: 0,
            ccs_heat_rate_penalty: 1.0,
            _action: 'add_plant',
            _year_online: yearOnline,
            _ccs_target_rate: 0,
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
        setStatus('Added "' + name + '" — click Recalculate to update charts');
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

        // Use requestAnimationFrame to let UI update before heavy compute
        requestAnimationFrame(function () {
            setTimeout(function () {
                try {
                    var t0 = performance.now();
                    var result = FleetDispatchEngine.computeFleetDispatch(allPlants, sweepData);
                    var elapsed = Math.round(performance.now() - t0);

                    // Build scenario name
                    var scenarioName = (els.nameInput && els.nameInput.value.trim()) || 'Custom';
                    var scenarioKey = 'custom_' + Date.now();

                    // Package result for fleet-scenarios.js
                    var scenarioData = {
                        description: scenarioName,
                        color: '#8B5CF6',
                        envelope: result.envelope,
                        plant_detail: result.plant_detail,
                        generation_by_fuel: result.generation_by_fuel,
                        emissions_by_fuel: result.emissions_by_fuel,
                        fleet_summary: result.fleet_summary
                    };

                    // Inject via API
                    if (window.FLEET_SCENARIOS_API) {
                        window.FLEET_SCENARIOS_API.addScenario(scenarioKey, scenarioName, scenarioData);
                        setStatus('Done in ' + elapsed + 'ms — "' + scenarioName + '" added to charts');
                    } else {
                        console.warn('FLEET_SCENARIOS_API not available');
                        setStatus('Computed in ' + elapsed + 'ms but chart API unavailable');
                    }
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

    // ── Save/Load Scenarios ──
    function saveScenario() {
        var name = (els.nameInput && els.nameInput.value.trim()) || '';
        if (!name) { alert('Please enter a scenario name'); return; }

        savedScenarios[name] = {
            timestamp: Date.now(),
            fleetMods: fleetPlants.filter(function (p) { return p._action; }).map(function (p) {
                return { _idx: p._idx, orispl: p.orispl, _action: p._action, _year_online: p._year_online, _ccs_target_rate: p._ccs_target_rate, capacity_mw: p.capacity_mw };
            }),
            addedPlants: JSON.parse(JSON.stringify(addedPlants))
        };

        try {
            localStorage.setItem('fleet_custom_scenarios', JSON.stringify(savedScenarios));
        } catch (e) { console.warn('localStorage save failed:', e); }

        renderSavedScenarios();
        setStatus('Saved "' + name + '"');
    }

    function loadScenario(name) {
        var saved = savedScenarios[name];
        if (!saved) return;

        // Reset fleet to base
        fleetPlants = JSON.parse(JSON.stringify(baseFleet));

        // Apply saved modifications
        (saved.fleetMods || []).forEach(function (mod) {
            var p = fleetPlants[mod._idx];
            if (p && p.orispl === mod.orispl) {
                p._action = mod._action;
                p._year_online = mod._year_online;
                p._ccs_target_rate = mod._ccs_target_rate;
                if (mod.capacity_mw != null) p.capacity_mw = mod.capacity_mw;
            }
        });

        // Restore added plants
        addedPlants = JSON.parse(JSON.stringify(saved.addedPlants || []));

        // Update name input
        if (els.nameInput) els.nameInput.value = name;

        renderFleetList();
        setStatus('Loaded "' + name + '" — click Recalculate to update charts');
    }

    function deleteScenario(name) {
        delete savedScenarios[name];
        try {
            localStorage.setItem('fleet_custom_scenarios', JSON.stringify(savedScenarios));
        } catch (e) {}
        renderSavedScenarios();
    }

    function renderSavedScenarios() {
        if (!els.savedList) return;
        var names = Object.keys(savedScenarios);
        if (!names.length) {
            els.savedList.innerHTML = '<div style="color:#9ca3af;font-size:0.82rem;padding:8px 0;">No saved scenarios yet</div>';
            return;
        }
        var html = '';
        names.forEach(function (name) {
            html += '<div class="sb-saved-item">';
            html += '<span>' + name + '</span>';
            html += '<div class="sb-saved-actions">';
            html += '<button class="sb-btn" data-load="' + name + '" style="padding:4px 10px;font-size:0.78rem;min-height:32px;">Load</button>';
            html += '<button class="sb-btn-danger" data-delete="' + name + '">✕</button>';
            html += '</div>';
            html += '</div>';
        });
        els.savedList.innerHTML = html;

        // Bind load/delete
        els.savedList.querySelectorAll('[data-load]').forEach(function (btn) {
            btn.addEventListener('click', function () { loadScenario(this.dataset.load); });
        });
        els.savedList.querySelectorAll('[data-delete]').forEach(function (btn) {
            btn.addEventListener('click', function () { deleteScenario(this.dataset.delete); });
        });
    }

    // ── Reset fleet to base ──
    function resetFleet() {
        fleetPlants = JSON.parse(JSON.stringify(baseFleet));
        addedPlants = [];
        renderFleetList();
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
        resetFleet: resetFleet
    };

})();
