/**
 * Market Simulation Screening Tool — Setup Page Logic
 * Handles form interactions, preset buttons, mode switching, and form submission.
 */

// Storage $/kW-yr → $/MWh LCOS conversion divisors
const STORAGE_DIVISORS = { battery: 1.241, battery8: 2.040, ldes: 0.500 };

function updateStorageLCOE(type) {
    const cost = parseFloat(document.getElementById(`cost_${type}`).value) || 0;
    const lcoe = (cost / STORAGE_DIVISORS[type]).toFixed(1);
    const el = document.getElementById(`lcoe_${type}_computed`);
    if (el) el.textContent = `≈ $${lcoe}/MWh`;
}

// ISO summary data (from pipeline_config)
const ISO_DATA = {
    CAISO: { demand_twh: 224.0, clean_pct: 48.5, fossil_mw: 47000, cap_market: 75 },
    ERCOT: { demand_twh: 488.0, clean_pct: 46.1, fossil_mw: 80000, cap_market: 0 },
    PJM:   { demand_twh: 843.3, clean_pct: 40.6, fossil_mw: 127800, cap_market: 120 },
    NYISO: { demand_twh: 151.6, clean_pct: 39.0, fossil_mw: 28000, cap_market: 85 },
    NEISO: { demand_twh: 115.3, clean_pct: 33.5, fossil_mw: 16000, cap_market: 55 },
    MISO:  { demand_twh: 660.0, clean_pct: 31.3, fossil_mw: 105000, cap_market: 25 },
    SPP:   { demand_twh: 296.0, clean_pct: 47.0, fossil_mw: 58000, cap_market: 0 },
};

const MODE_DESCRIPTIONS = {
    trajectory5: "Multi-year market trajectory to 2060 at 5-year intervals. Wright's Law learning curves reduce costs each year. Shows when the market flips for each resource type.",
    trajectory1: "Multi-year market trajectory to 2060 at annual resolution. Same model as 5-year but with single-year granularity. Detailed results downloadable as CSV.",
    sweep: "Parametric market reference sweep at 5-year intervals: 3 demand × 5 price × 3 PPA × 3 gas friction × 3 queue speed × 3 new-build fossil cost = 1,215 scenarios. Upload custom CSV templates to override defaults."
};

// Mode → year step mapping
const MODE_STEP = { trajectory5: 5, trajectory1: 1, sweep: 5 };

// ── Mode switching ──
function updateSweepModeUI(mode) {
    const isSweep = (mode === 'sweep');
    // Show/hide all parameter sections
    document.querySelectorAll('[data-sweep-hide]').forEach(el => {
        el.style.display = isSweep ? 'none' : '';
    });
    // Show/hide sweep info panel
    const sweepPanel = document.getElementById('sweepInfoPanel');
    if (sweepPanel) sweepPanel.style.display = isSweep ? '' : 'none';
    // Show/hide sweep cache info
    const sweepCacheInfo = document.getElementById('sweepCacheInfo');
    if (sweepCacheInfo) sweepCacheInfo.style.display = isSweep ? '' : 'none';
    // Hide fleet CTA + CSV section + nuclear retirement in sweep mode
    const fleetCTA = document.getElementById('fleetCTA');
    if (fleetCTA) fleetCTA.style.display = isSweep ? 'none' : '';
    const csvSection = document.getElementById('csvConfigSection');
    if (csvSection) csvSection.style.display = isSweep ? 'none' : '';
    const nucCard = document.getElementById('nuclearRetirementCard');
    if (nucCard) nucCard.style.display = isSweep ? 'none' : '';
    const fleetCard = document.getElementById('fleetConfigCard');
    if (fleetCard) fleetCard.style.display = isSweep ? 'none' : '';
    // Update submit button text
    const submitText = document.querySelector('.submit-text');
    if (submitText) {
        submitText.textContent = isSweep
            ? 'Load Cached Sweep Results'
            : 'Run Market Simulation Screening';
    }
    // Check cache availability when switching to sweep
    if (isSweep) checkSweepCacheStatus();
}

async function checkSweepCacheStatus() {
    const statusEl = document.getElementById('sweepCacheStatus');
    if (!statusEl) return;
    try {
        const resp = await fetch('/api/sweep-cached/status');
        const data = await resp.json();
        if (data.available) {
            const sizeMB = data.json_size_mb || '?';
            statusEl.innerHTML = `<span style="color:#22C55E;font-weight:600;">✓ Cached results available</span>` +
                ` (${data.total_scenarios} scenarios × ${data.isos.length} ISOs × ${data.years.length} years` +
                ` = ${data.total_scenarios * data.isos.length * data.years.length} results, ${sizeMB} MB)`;
            statusEl.style.borderColor = '#22C55E33';
            statusEl.style.background = '#22C55E08';
        } else {
            statusEl.innerHTML = `<span style="color:#EF4444;font-weight:600;">✗ No cached results</span>` +
                ` — Run the GitHub Actions workflow <em>"Market Simulator: 405-Scenario Sweep"</em> first, ` +
                `or use Trajectory mode to run a single custom scenario.`;
            statusEl.style.borderColor = '#EF444433';
            statusEl.style.background = '#EF444408';
        }
    } catch (e) {
        statusEl.innerHTML = `<span style="color:#F59E0B;">⚠ Could not check cache status</span> (${e.message})`;
        statusEl.style.borderColor = '#F59E0B33';
        statusEl.style.background = '#F59E0B08';
    }
}

document.querySelectorAll('.mode-toggle .toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.mode-toggle .toggle-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const mode = btn.dataset.mode;
        document.getElementById('modeDescription').textContent = MODE_DESCRIPTIONS[mode];
        updateYearCountHint();
        updateSweepModeUI(mode);
    });
});

// ── Year range controls ──
function updateYearCountHint() {
    const start = parseInt(document.getElementById('startYear')?.value || 2025);
    const end = parseInt(document.getElementById('endYear')?.value || 2060);
    const mode = document.querySelector('.mode-toggle .toggle-btn.active')?.dataset.mode || 'trajectory5';
    const step = MODE_STEP[mode] || 5;
    const count = Math.floor((end - start) / step) + 1;
    const hint = document.getElementById('yearCountHint');
    if (hint) {
        hint.textContent = `${count} simulation year${count !== 1 ? 's' : ''} (${step === 1 ? 'annual' : '5-yr intervals'})`;
        if (count > 20) hint.textContent += ' (may be slow for sweeps)';
    }
}

document.getElementById('startYear')?.addEventListener('change', updateYearCountHint);
document.getElementById('endYear')?.addEventListener('change', updateYearCountHint);

// ── Data tier status (fetched once) ──
let _dataTierCache = null;

async function fetchDataTiers() {
    if (_dataTierCache) return _dataTierCache;
    try {
        const resp = await fetch('/api/data-status');
        if (resp.ok) {
            _dataTierCache = await resp.json();
            return _dataTierCache;
        }
    } catch (e) { /* ignore — indicator will stay empty */ }
    return null;
}

function renderDataTierIndicator(iso) {
    const el = document.getElementById('dataTierIndicator');
    if (!el || !_dataTierCache) { if (el) el.innerHTML = ''; return; }

    const tiers = (_dataTierCache.tiers || {})[iso];
    if (!tiers) { el.innerHTML = ''; return; }

    const items = [];
    const dot = (color) => `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color};margin-right:4px;"></span>`;

    // Resource mix
    if (tiers.resource_mix === 'parquet') {
        items.push(`${dot('#22C55E')}Resource mix: Physics data`);
    } else {
        items.push(`${dot('#EF4444')}Resource mix: Synthetic`);
    }

    // Interchange
    if (tiers.interchange === 'eia_930') {
        items.push(`${dot('#22C55E')}Interchange: EIA-930`);
    } else {
        items.push(`${dot('#F59E0B')}Interchange: Not loaded`);
    }

    // Zonal config
    if (tiers.zonal_config === 'validated') {
        items.push(`${dot('#22C55E')}Zonal config: Validated`);
    } else {
        items.push(`${dot('#F59E0B')}Zonal config: Hardcoded`);
    }

    // Fleet data
    if (tiers.fleet_data === 'plant_level') {
        items.push(`${dot('#22C55E')}Fleet data: Plant-level`);
    } else {
        items.push(`${dot('#0EA5E9')}Fleet data: Aggregated`);
    }

    // DR params
    if (tiers.dr_params === 'calibrated') {
        items.push(`${dot('#22C55E')}DR params: Calibrated`);
    } else {
        items.push(`${dot('#0EA5E9')}DR params: Default`);
    }

    const allGreen = tiers.resource_mix === 'parquet' && tiers.interchange === 'eia_930'
        && tiers.fleet_data === 'plant_level';
    const headerColor = tiers.resource_mix === 'synthetic' ? '#EF4444' : (allGreen ? '#22C55E' : '#F59E0B');
    const headerLabel = tiers.resource_mix === 'synthetic' ? 'Synthetic Data' : (allGreen ? 'Full Data' : 'Partial Data');

    el.innerHTML = `<div style="padding:0.75rem 1rem;border-radius:8px;border:1px solid ${headerColor}33;background:${headerColor}08;font-size:0.8rem;">
        <div style="font-weight:600;margin-bottom:0.5rem;color:${headerColor};">${headerLabel}</div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:0.25rem 1rem;color:#4B5563;">
            ${items.map(i => `<span>${i}</span>`).join('')}
        </div>
    </div>`;
}

// ── ISO selection — always single-select (all modes) ──
document.querySelectorAll('.iso-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        // Single select in all modes
        document.querySelectorAll('.iso-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        updateISOSummary();
        updateGeothermalVisibility();
        renderDataTierIndicator(btn.dataset.iso);
    });
});

function updateISOSummary() {
    const container = document.getElementById('isoSummary');
    const selected = Array.from(document.querySelectorAll('.iso-btn.active')).map(b => b.dataset.iso);
    container.innerHTML = selected.map(iso => {
        const d = ISO_DATA[iso];
        return `<div class="iso-summary-card">
            <div class="iso-name">${iso}</div>
            <div class="iso-stat">${d.demand_twh} TWh demand · ${d.clean_pct}% clean</div>
            <div class="iso-stat">${(d.fossil_mw/1000).toFixed(0)} GW fossil · Cap mkt: $${d.cap_market}/kW-yr</div>
        </div>`;
    }).join('');
}

function updateGeothermalVisibility() {
    const selected = Array.from(document.querySelectorAll('.iso-btn.active')).map(b => b.dataset.iso);
    document.getElementById('geoGroup').style.display = selected.includes('CAISO') ? '' : 'none';
}

// ── Preset buttons (L/M/H) ──
document.querySelectorAll('.preset-btn, .preset-btn-wide').forEach(btn => {
    btn.addEventListener('click', () => {
        const target = btn.dataset.target;
        const value = parseFloat(btn.dataset.value);
        document.getElementById(target).value = value;

        // Update active state within sibling buttons
        const parent = btn.parentElement;
        parent.querySelectorAll('.preset-btn, .preset-btn-wide').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Update storage LCOE readouts when preset buttons are clicked
        if (target === 'cost_battery') updateStorageLCOE('battery');
        else if (target === 'cost_battery8') updateStorageLCOE('battery8');
        else if (target === 'cost_ldes') updateStorageLCOE('ldes');
    });
});

// When user manually edits an input, clear preset active states
document.querySelectorAll('.input-with-presets input, .carbon-controls input').forEach(input => {
    input.addEventListener('input', () => {
        const presets = input.closest('.input-with-presets, .carbon-controls');
        if (presets) {
            presets.querySelectorAll('.preset-btn, .preset-btn-wide').forEach(b => {
                const val = parseFloat(b.dataset.value);
                if (Math.abs(val - parseFloat(input.value)) < 0.01) {
                    b.classList.add('active');
                } else {
                    b.classList.remove('active');
                }
            });
        }
    });
});

// ── Toggle button groups ──
document.querySelectorAll('.toggle-btn-group').forEach(group => {
    group.querySelectorAll('.toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            group.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });
});

// ── Collapsible sections ──
document.querySelectorAll('.collapsible-header').forEach(header => {
    header.addEventListener('click', () => {
        header.closest('.collapsible').classList.toggle('open');
    });
});

// ── Learning curves toggle ──
document.querySelectorAll('#learningToggle .toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const isOn = btn.dataset.value === 'On';
        document.getElementById('learningSpeedGroup').style.display = isOn ? '' : 'none';
    });
});

// ── Form validation ──
function validateForm() {
    const errors = [];
    const warnings = [];

    // Required numeric fields — reject non-numeric or empty
    const requiredNumeric = [
        { id: 'fuel_gas', label: 'Natural Gas price', min: 0 },
        { id: 'fuel_coal', label: 'Coal price', min: 0 },
        { id: 'fuel_oil', label: 'Oil price', min: 0 },
        { id: 'carbon_price', label: 'Carbon price', min: 0 },
        { id: 'nuclear_retirement', label: 'Nuclear retirement threshold', min: 0 },
        { id: 'lcoe_solar', label: 'Solar LCOE', min: 0 },
        { id: 'lcoe_wind', label: 'Wind LCOE', min: 0 },
        { id: 'lcoe_offshore', label: 'Offshore Wind LCOE', min: 0 },
        { id: 'lcoe_nuclear', label: 'Nuclear LCOE', min: 0 },
        { id: 'lcoe_ccs', label: 'CCS-CCGT LCOE', min: 0 },
        { id: 'cost_battery', label: 'Battery 4hr cost', min: 0 },
        { id: 'cost_battery8', label: 'Battery 8hr cost', min: 0 },
        { id: 'cost_ldes', label: 'LDES cost', min: 0 },
    ];

    for (const field of requiredNumeric) {
        const el = document.getElementById(field.id);
        if (!el) continue;
        const val = el.value.trim();
        if (val === '') {
            errors.push(`${field.label} is required`);
            el.style.borderColor = 'var(--danger, #dc3545)';
        } else if (isNaN(parseFloat(val))) {
            errors.push(`${field.label} must be a number`);
            el.style.borderColor = 'var(--danger, #dc3545)';
        } else if (field.min !== undefined && parseFloat(val) < field.min) {
            errors.push(`${field.label} cannot be negative`);
            el.style.borderColor = 'var(--danger, #dc3545)';
        } else {
            el.style.borderColor = '';
        }
    }

    // Warn on extreme values (allow but flag)
    const demandToggle = document.querySelector('#demandToggle .toggle-btn.active');
    if (demandToggle && demandToggle.dataset.value === 'Custom') {
        const customVal = parseFloat(document.getElementById('custom_demand_pct')?.value);
        if (customVal > 7.5) {
            warnings.push(`Demand growth ${customVal}% exceeds 7.5% — backend will cap at 7.5%`);
        }
    }

    const gasPrice = parseFloat(document.getElementById('fuel_gas')?.value);
    if (gasPrice > 15) warnings.push(`Gas price $${gasPrice}/MMBtu is unusually high`);

    const carbonPrice = parseFloat(document.getElementById('carbon_price')?.value);
    if (carbonPrice > 300) warnings.push(`Carbon price $${carbonPrice}/ton is unusually high`);

    return { errors, warnings };
}

// ── Form submission ──
document.getElementById('simulationForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const submitBtn = document.getElementById('submitBtn');
    const submitText = submitBtn.querySelector('.submit-text');
    const submitSpinner = submitBtn.querySelector('.submit-spinner');
    const status = document.getElementById('submitStatus');

    const activeMode = document.querySelector('.mode-toggle .toggle-btn.active')?.dataset.mode || 'trajectory5';

    // Skip validation and form collection for cached sweep mode
    let params = {};
    if (activeMode !== 'sweep') {
        const { errors, warnings } = validateForm();
        if (errors.length > 0) {
            status.className = 'submit-status error';
            status.textContent = `Validation errors: ${errors.join('; ')}`;
            return;
        }
        if (warnings.length > 0) {
            status.className = 'submit-status';
            status.style.color = '#D97706';
            status.textContent = `Warning: ${warnings.join('; ')} — submitting anyway`;
        }
        params = collectFormData();
    }

    // UI feedback
    submitBtn.disabled = true;
    submitText.style.display = 'none';
    submitSpinner.style.display = '';
    status.textContent = '';
    status.className = 'submit-status';

    try {
        const mode = document.querySelector('.mode-toggle .toggle-btn.active').dataset.mode;

        if (mode === 'sweep') {
            // ── Cached sweep mode — load from pre-computed results ──
            const selectedISO = document.querySelector('.iso-btn.active')?.dataset.iso || '';
            const isoParam = selectedISO ? `?iso=${selectedISO}` : '';

            status.textContent = 'Loading cached sweep results...';

            // Load aggregates
            const aggResp = await fetch(`/api/sweep-cached/aggregates${isoParam}`);
            if (!aggResp.ok) {
                const err = await aggResp.json();
                throw new Error(err.detail || `HTTP ${aggResp.status}`);
            }
            const aggregates = await aggResp.json();

            // Load full results (filtered by ISO for reasonable size)
            const resResp = await fetch(`/api/sweep-cached/results${isoParam}`);
            if (!resResp.ok) {
                const err = await resResp.json();
                throw new Error(err.detail || `HTTP ${resResp.status}`);
            }
            const fullResults = await resResp.json();

            // Store in sessionStorage for results page
            sessionStorage.setItem('sweepResult', JSON.stringify({
                scenario_count: fullResults.scenario_count,
                aggregates: aggregates,
                results: fullResults.results,
                iso: selectedISO || 'ALL',
                cached: true,
            }));
            sessionStorage.setItem('simulationParams', JSON.stringify({
                mode: 'sweep',
                iso: selectedISO || 'ALL',
                cached: true,
            }));

            status.className = 'submit-status success';
            status.textContent = `Loaded ${fullResults.scenario_count} cached scenarios. Redirecting...`;
            setTimeout(() => { window.location.href = '/results'; }, 500);

        } else {
            // ── Single scenario mode — run live ──
            const endpoint = '/api/simulate';
            if (mode === 'trajectory5') params.mode = 'trajectory';
            else if (mode === 'trajectory1') params.mode = 'trajectory';
            else params.mode = mode;

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params),
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || `HTTP ${response.status}`);
            }

            const result = await response.json();
            sessionStorage.setItem('simulationResult', JSON.stringify(result));
            sessionStorage.setItem('simulationParams', JSON.stringify(params));
            status.className = 'submit-status success';
            status.textContent = 'Simulation complete! Redirecting to results...';
            setTimeout(() => { window.location.href = '/results'; }, 500);
        }
    } catch (err) {
        status.className = 'submit-status error';
        status.textContent = `Error: ${err.message}`;
    } finally {
        submitBtn.disabled = false;
        submitText.style.display = '';
        submitSpinner.style.display = 'none';
    }
});

function collectFormData() {
    const mode = document.querySelector('.mode-toggle .toggle-btn.active').dataset.mode;
    const selectedISO = document.querySelector('.iso-btn.active')?.dataset.iso || 'CAISO';

    // Build per-resource TX overrides (only non-empty values)
    const txOverrides = {};
    ['solar', 'wind', 'offshore_wind', 'nuclear', 'ccs_ccgt', 'geothermal'].forEach(res => {
        const el = document.getElementById(`tx_override_${res}`);
        if (el && el.value !== '') {
            txOverrides[res] = parseFloat(el.value);
        }
    });

    const params = {
        mode: mode,
        iso: selectedISO,
        fuel_prices: {
            gas: parseFloat(document.getElementById('fuel_gas').value),
            coal: parseFloat(document.getElementById('fuel_coal').value),
            oil: parseFloat(document.getElementById('fuel_oil').value),
        },
        carbon_price: parseFloat(document.getElementById('carbon_price').value),
        emission_prices: {
            nox: parseFloat(document.getElementById('nox_price').value) || 0,
            sox: parseFloat(document.getElementById('sox_price').value) || 0,
        },
        emission_limits: {
            nox_limit: document.getElementById('nox_limit').value ?
                parseFloat(document.getElementById('nox_limit').value) : null,
            sox_limit: document.getElementById('sox_limit').value ?
                parseFloat(document.getElementById('sox_limit').value) : null,
        },
        heat_rates: {
            coal_steam: parseFloat(document.getElementById('hr_coal_steam').value),
            gas_ccgt: parseFloat(document.getElementById('hr_gas_ccgt').value),
            gas_ct: parseFloat(document.getElementById('hr_gas_ct').value),
            oil_ct: parseFloat(document.getElementById('hr_oil_ct').value),
            new_gas_ccgt: parseFloat(document.getElementById('hr_new_gas_ccgt').value),
            new_gas_ct: parseFloat(document.getElementById('hr_new_gas_ct').value),
            new_coal: parseFloat(document.getElementById('hr_new_coal').value),
        },
        vom: {
            coal_steam: parseFloat(document.getElementById('vom_coal_steam').value),
            gas_ccgt: parseFloat(document.getElementById('vom_gas_ccgt').value),
            gas_ct: parseFloat(document.getElementById('vom_gas_ct').value),
            oil_ct: parseFloat(document.getElementById('vom_oil_ct').value),
        },
        clean_lcoes: {
            solar: parseFloat(document.getElementById('lcoe_solar').value),
            wind: parseFloat(document.getElementById('lcoe_wind').value),
            offshore_wind: parseFloat(document.getElementById('lcoe_offshore').value),
            nuclear: parseFloat(document.getElementById('lcoe_nuclear').value),
            ccs_ccgt: parseFloat(document.getElementById('lcoe_ccs').value),
            geothermal: parseFloat(document.getElementById('lcoe_geo').value || 55),
        },
        fossil_lcoes: {
            gas_ccgt: parseFloat(document.getElementById('new_gas_ccgt_lcoe').value),
            gas_ct: parseFloat(document.getElementById('new_gas_ct_lcoe').value),
            coal: parseFloat(document.getElementById('new_coal_lcoe').value),
        },
        incentives: {
            ptc_wind: parseFloat(document.getElementById('ptc_wind').value) || 0,
            ptc_solar: parseFloat(document.getElementById('ptc_solar').value) || 0,
            ptc_nuclear_new: parseFloat(document.getElementById('ptc_nuclear_new').value) || 0,
            ptc_45u_max: parseFloat(document.getElementById('ptc_45u_max').value) || 0,
            ptc_45u_floor: parseFloat(document.getElementById('ptc_45u_floor').value) || 0,
            ptc_45u_floor_escalation: parseFloat(document.getElementById('ptc_45u_floor_escalation').value) || 0,
            ptc_45u_sunset_year: parseInt(document.getElementById('ptc_45u_sunset_year').value) || 2032,
            itc_pct: parseFloat(document.getElementById('itc_pct').value) || 0,
            rec_price: document.getElementById('rec_price').value ?
                parseFloat(document.getElementById('rec_price').value) : null,
        },
        storage_costs: {
            battery: parseFloat(document.getElementById('cost_battery').value),
            battery8: parseFloat(document.getElementById('cost_battery8').value),
            ldes: parseFloat(document.getElementById('cost_ldes').value),
        },
        capacity_market_price: document.getElementById('capacity_market').value ?
            parseFloat(document.getElementById('capacity_market').value) : null,
        wholesale_price_override: document.getElementById('wholesale_override').value ?
            parseFloat(document.getElementById('wholesale_override').value) : null,
        transmission_level: document.querySelector('#txToggle .toggle-btn.active')?.dataset.value || 'Medium',
        tx_overrides: txOverrides,
        q45: document.querySelector('#q45Toggle .toggle-btn.active')?.dataset.value === '1',
        ccs_credit_override: document.getElementById('ccs_credit_override').value ?
            parseFloat(document.getElementById('ccs_credit_override').value) : null,
        demand_growth: (() => {
            const sel = document.querySelector('#demandToggle .toggle-btn.active')?.dataset.value || 'Medium';
            if (sel === 'Custom') {
                return parseFloat(document.getElementById('custom_demand_pct')?.value) || 1.5;
            }
            return sel;
        })(),
        ppa_level: document.querySelector('#ppaToggle .toggle-btn.active')?.dataset.value || 'Medium',
        gas_friction: document.querySelector('#gasFrictionToggle .toggle-btn.active')?.dataset.value || 'Medium',
        new_fossil_cost_level: document.querySelector('#newFossilCostToggle .toggle-btn.active')?.dataset.value || 'Medium',
        new_fossil_enabled: document.querySelector('#newFossilEnabledToggle .toggle-btn.active')?.dataset.value === 'On',
        new_fossil_min_cf_override: {
            gas_ccgt: (parseFloat(document.getElementById('min_cf_ccgt')?.value) || 30) / 100.0,
            gas_ct: (parseFloat(document.getElementById('min_cf_ct')?.value) || 5) / 100.0,
            coal: (parseFloat(document.getElementById('min_cf_coal')?.value) || 60) / 100.0,
        },
        interchange_enabled: document.querySelector('#interchangeToggle .toggle-btn.active')?.dataset.value === 'On',
        dr_level: document.querySelector('#drToggle .toggle-btn.active')?.dataset.value || 'Off',
        scarcity_mode: document.querySelector('#scarcityModeToggle .toggle-btn.active')?.dataset.value || 'ordc',
        nuclear_retirement_threshold: parseFloat(document.getElementById('nuclear_retirement').value),
        custom_overrides: {
            fuel: document.getElementById('custom_fuel_toggle')?.checked || false,
            lmp: document.getElementById('custom_lmp_toggle')?.checked || false,
            capacity: document.getElementById('custom_capacity_toggle')?.checked || false,
            rec: document.getElementById('custom_rec_toggle')?.checked || false,
        },
    };

    // Fleet overrides from fleet-config page (stored in localStorage)
    try {
        const saved = localStorage.getItem('fleet_overrides');
        if (saved && saved !== '{}') {
            params.fleet_overrides = JSON.parse(saved);
        }
    } catch (e) { /* ignore parse errors */ }

    // Trajectory params (all modes are trajectory-based now)
    const activeMode = document.querySelector('.mode-toggle .toggle-btn.active')?.dataset.mode || 'trajectory5';
    params.start_year = parseInt(document.getElementById('startYear')?.value || 2025);
    params.end_year = parseInt(document.getElementById('endYear')?.value || 2060);
    params.year_step = MODE_STEP[activeMode] || 5;
    params.learning_curves = document.querySelector('#learningToggle .toggle-btn.active')?.dataset.value === 'On';
    params.learning_speed = document.querySelector('#learningSpeedToggle .toggle-btn.active')?.dataset.value || 'Medium';
    params.queue_cap_level = document.querySelector('#queueCapToggle .toggle-btn.active')?.dataset.value || 'Medium';
    params.tech_differentiated_queue = (document.querySelector('#queueModelToggle .toggle-btn.active')?.dataset.value || 'tech') === 'tech';
    const queueOverride = document.getElementById('queue_cap_override')?.value;
    if (queueOverride && queueOverride !== '') {
        params.queue_cap_override_gw = parseFloat(queueOverride);
    }

    return params;
}

async function pollSweepStatus(jobId) {
    const status = document.getElementById('submitStatus');
    let attempts = 0;
    const maxAttempts = 600; // 10 min at 1s intervals

    const poll = async () => {
        attempts++;
        try {
            const resp = await fetch(`/api/sweep/${jobId}`);
            const data = await resp.json();

            if (data.status === 'completed') {
                sessionStorage.setItem('sweepResult', JSON.stringify(data.results));
                status.className = 'submit-status success';
                status.textContent = `Sweep complete! ${data.scenarios_completed} scenarios. Redirecting...`;
                setTimeout(() => { window.location.href = '/results'; }, 500);
                return;
            } else if (data.status === 'failed') {
                status.className = 'submit-status error';
                status.textContent = `Sweep failed: ${data.error}`;
                return;
            }

            status.textContent = `Running... ${data.scenarios_completed || 0}/${data.scenarios_total || '?'} scenarios`;

            if (attempts < maxAttempts) {
                setTimeout(poll, 1000);
            } else {
                status.className = 'submit-status error';
                status.textContent = 'Sweep timed out. Check server logs.';
            }
        } catch (err) {
            status.className = 'submit-status error';
            status.textContent = `Poll error: ${err.message}`;
        }
    };

    poll();
}

// ── Custom demand growth toggle ──
document.querySelectorAll('#demandToggle .toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const isCustom = btn.dataset.value === 'Custom';
        const customInput = document.getElementById('customDemandInput');
        if (customInput) {
            customInput.classList.toggle('visible', isCustom);
        }
    });
});

// ── Custom file status checking ──
async function checkCustomInputStatus() {
    try {
        const resp = await fetch('/api/custom-input-status');
        if (!resp.ok) return;
        const status = await resp.json();
        const fileMap = {
            'fuel': 'customFuelStatus',
            'lmp': 'customLmpStatus',
            'capacity': 'customCapacityStatus',
            'rec': 'customRecStatus',
        };
        for (const [key, elementId] of Object.entries(fileMap)) {
            const el = document.getElementById(elementId);
            if (!el) continue;
            const fileStatus = status[key];
            if (!fileStatus) {
                el.textContent = '';
                continue;
            }
            if (fileStatus.found && fileStatus.valid) {
                el.textContent = `✓ File found and valid (${fileStatus.rows} rows)`;
                el.className = 'custom-file-status found';
            } else if (fileStatus.found && !fileStatus.valid) {
                el.textContent = `✗ File found but invalid: ${fileStatus.error}`;
                el.className = 'custom-file-status invalid';
            } else {
                el.textContent = 'No custom file found — using model defaults';
                el.className = 'custom-file-status missing';
            }
        }
    } catch (e) {
        // Silently fail — custom inputs are optional
    }
}

// Check custom file toggles and update status on toggle change
document.querySelectorAll('#custom_fuel_toggle, #custom_lmp_toggle, #custom_capacity_toggle, #custom_rec_toggle').forEach(toggle => {
    toggle.addEventListener('change', () => checkCustomInputStatus());
});

// ── Fleet override status display ──
function updateFleetStatus() {
    const saved = localStorage.getItem('fleet_overrides');
    const statusEl = document.getElementById('fleetStatusText');
    if (!statusEl) return;

    if (!saved || saved === '{}') {
        statusEl.textContent = 'Using defaults (all plants at baseline status)';
        statusEl.style.color = 'var(--text-muted)';
        return;
    }

    try {
        const overrides = JSON.parse(saved);
        const count = Object.keys(overrides).length;
        if (count === 0) {
            statusEl.textContent = 'Using defaults (all plants at baseline status)';
            statusEl.style.color = 'var(--text-muted)';
            return;
        }
        const retired = Object.values(overrides).filter(v => v === 'Retired').length;
        const ccs = Object.values(overrides).filter(v => v === 'CCS Retrofit').length;
        const parts = [];
        if (retired > 0) parts.push(`${retired} retired`);
        if (ccs > 0) parts.push(`${ccs} CCS retrofit`);
        statusEl.innerHTML = `<span style="color: #22C55E; font-weight: 600;">✓</span> ${count} plant${count > 1 ? 's' : ''} modified (${parts.join(', ')})`;
        statusEl.style.color = 'var(--navy)';
    } catch (e) {
        statusEl.textContent = 'Using defaults (all plants at baseline status)';
    }
}

// ── CSV Upload/Download ──

// Toggle button mapping: CSV parameter → {selector, attribute, matchKey}
const TOGGLE_MAP = {
    iso:                  { selector: '.iso-btn', attr: 'data-iso' },
    mode:                 { selector: '.mode-toggle .toggle-btn', attr: 'data-mode' },
    transmission_level:   { selector: '#txToggle .toggle-btn', attr: 'data-value' },
    q45:                  { selector: '#q45Toggle .toggle-btn', attr: 'data-value', map: { 'On': '1', 'Off': '0' } },
    ppa_level:            { selector: '#ppaToggle .toggle-btn', attr: 'data-value' },
    demand_growth:        { selector: '#demandToggle .toggle-btn', attr: 'data-value' },
    gas_friction:         { selector: '#gasFrictionToggle .toggle-btn', attr: 'data-value' },
    new_fossil_cost_level: { selector: '#newFossilCostToggle .toggle-btn', attr: 'data-value' },
    new_fossil_enabled:   { selector: '#newFossilEnabledToggle .toggle-btn', attr: 'data-value' },
    interchange_enabled:  { selector: '#interchangeToggle .toggle-btn', attr: 'data-value' },
    dr_level:             { selector: '#drToggle .toggle-btn', attr: 'data-value' },
    scarcity_mode:        { selector: '#scarcityModeToggle .toggle-btn', attr: 'data-value' },
    learning_curves:      { selector: '#learningToggle .toggle-btn', attr: 'data-value' },
    learning_speed:       { selector: '#learningSpeedToggle .toggle-btn', attr: 'data-value' },
    queue_cap_level:      { selector: '#queueCapToggle .toggle-btn', attr: 'data-value' },
    tech_differentiated_queue: { selector: '#queueModelToggle .toggle-btn', attr: 'data-value', map: { 'tech': '1', 'uniform': '0' } },
};

// Simple number input mapping: CSV parameter → DOM element ID
const INPUT_MAP = {
    fuel_gas: 'fuel_gas', fuel_coal: 'fuel_coal', fuel_oil: 'fuel_oil',
    carbon_price: 'carbon_price', nox_price: 'nox_price', sox_price: 'sox_price',
    nox_limit: 'nox_limit', sox_limit: 'sox_limit',
    hr_coal_steam: 'hr_coal_steam', hr_gas_ccgt: 'hr_gas_ccgt',
    hr_gas_ct: 'hr_gas_ct', hr_oil_ct: 'hr_oil_ct',
    hr_new_gas_ccgt: 'hr_new_gas_ccgt', hr_new_gas_ct: 'hr_new_gas_ct', hr_new_coal: 'hr_new_coal',
    vom_coal_steam: 'vom_coal_steam', vom_gas_ccgt: 'vom_gas_ccgt',
    vom_gas_ct: 'vom_gas_ct', vom_oil_ct: 'vom_oil_ct',
    lcoe_solar: 'lcoe_solar', lcoe_wind: 'lcoe_wind', lcoe_offshore: 'lcoe_offshore',
    lcoe_nuclear: 'lcoe_nuclear', lcoe_ccs: 'lcoe_ccs', lcoe_geo: 'lcoe_geo',
    new_gas_ccgt_lcoe: 'new_gas_ccgt_lcoe', new_gas_ct_lcoe: 'new_gas_ct_lcoe',
    new_coal_lcoe: 'new_coal_lcoe',
    cost_battery: 'cost_battery', cost_battery8: 'cost_battery8', cost_ldes: 'cost_ldes',
    tx_override_solar: 'tx_override_solar', tx_override_wind: 'tx_override_wind',
    tx_override_offshore_wind: 'tx_override_offshore_wind', tx_override_nuclear: 'tx_override_nuclear',
    tx_override_ccs_ccgt: 'tx_override_ccs_ccgt', tx_override_geothermal: 'tx_override_geothermal',
    ptc_wind: 'ptc_wind', ptc_solar: 'ptc_solar', ptc_nuclear_new: 'ptc_nuclear_new',
    ptc_45u_max: 'ptc_45u_max', ptc_45u_floor: 'ptc_45u_floor',
    ptc_45u_floor_escalation: 'ptc_45u_floor_escalation', ptc_45u_sunset_year: 'ptc_45u_sunset_year',
    itc_pct: 'itc_pct', rec_price: 'rec_price',
    ccs_credit_override: 'ccs_credit_override',
    capacity_market: 'capacity_market', wholesale_override: 'wholesale_override',
    custom_demand_pct: 'custom_demand_pct',
    nuclear_retirement: 'nuclear_retirement',
    queue_cap_override: 'queue_cap_override',
    min_cf_ccgt: 'min_cf_ccgt', min_cf_ct: 'min_cf_ct', min_cf_coal: 'min_cf_coal',
};

// Select (dropdown) mapping
const SELECT_MAP = {
    start_year: 'startYear',
    end_year: 'endYear',
};

function parseCSV(text) {
    const lines = text.split(/\r?\n/).filter(l => l.trim());
    if (lines.length < 2) return {};
    // Find column indices from header
    const header = lines[0].split(',').map(h => h.trim().toLowerCase());
    const paramIdx = header.indexOf('parameter');
    const valueIdx = header.indexOf('value');
    if (paramIdx < 0 || valueIdx < 0) return {};

    const result = {};
    for (let i = 1; i < lines.length; i++) {
        // Simple CSV parse (handles quoted fields with commas)
        const cols = [];
        let current = '';
        let inQuotes = false;
        for (const ch of lines[i]) {
            if (ch === '"') { inQuotes = !inQuotes; continue; }
            if (ch === ',' && !inQuotes) { cols.push(current.trim()); current = ''; continue; }
            current += ch;
        }
        cols.push(current.trim());
        const param = cols[paramIdx];
        const value = cols[valueIdx];
        if (param) result[param] = value;
    }
    return result;
}

function applyCSVConfig(config) {
    let applied = 0;
    let skipped = 0;

    for (const [param, value] of Object.entries(config)) {
        // Toggle buttons
        if (TOGGLE_MAP[param]) {
            const { selector, attr, map } = TOGGLE_MAP[param];
            const matchValue = map ? (map[value] || value) : value;
            const buttons = document.querySelectorAll(selector);
            let found = false;
            buttons.forEach(btn => {
                if (btn.getAttribute(attr) === matchValue) {
                    // Deactivate siblings
                    buttons.forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    btn.click(); // Trigger any event listeners
                    found = true;
                }
            });
            if (found) applied++;
            else skipped++;
            continue;
        }

        // Number/text inputs
        if (INPUT_MAP[param]) {
            const el = document.getElementById(INPUT_MAP[param]);
            if (el) {
                el.value = value; // Empty string for blank values is correct
                el.dispatchEvent(new Event('input', { bubbles: true }));
                applied++;
            } else {
                skipped++;
            }
            continue;
        }

        // Select dropdowns
        if (SELECT_MAP[param]) {
            const el = document.getElementById(SELECT_MAP[param]);
            if (el) {
                el.value = value;
                el.dispatchEvent(new Event('change', { bubbles: true }));
                applied++;
            } else {
                skipped++;
            }
            continue;
        }
    }

    // Update computed fields
    ['battery', 'battery8', 'ldes'].forEach(t => updateStorageLCOE(t));
    updateISOSummary();
    updateGeothermalVisibility();
    updateYearCountHint();

    return { applied, skipped };
}

// File upload handler
document.getElementById('csvUploadInput')?.addEventListener('change', (e) => {
    const file = e.target.files[0];
    const status = document.getElementById('csvUploadStatus');
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (ev) => {
        try {
            const config = parseCSV(ev.target.result);
            const count = Object.keys(config).length;
            if (count === 0) {
                status.textContent = '✗ No valid parameters found in CSV';
                status.style.color = '#DC2626';
                return;
            }
            const { applied, skipped } = applyCSVConfig(config);
            status.textContent = `✓ ${applied} fields populated` + (skipped > 0 ? ` (${skipped} skipped)` : '');
            status.style.color = '#22C55E';
        } catch (err) {
            status.textContent = `✗ Parse error: ${err.message}`;
            status.style.color = '#DC2626';
        }
    };
    reader.readAsText(file);
    // Reset so same file can be re-uploaded
    e.target.value = '';
});

// ── Initialize ──
updateISOSummary();
updateGeothermalVisibility();
checkCustomInputStatus();
updateFleetStatus();
fetchDataTiers().then(() => {
    const activeISO = document.querySelector('.iso-btn.active')?.dataset.iso || 'CAISO';
    renderDataTierIndicator(activeISO);
});
