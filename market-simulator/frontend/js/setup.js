/**
 * Market Simulator — Setup Page Logic
 * Handles form interactions, preset buttons, mode switching, and form submission.
 */

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
    snapshot: "One point-in-time simulation with your exact cost inputs. No learning curves, no year progression. Fast results.",
    trajectory: "Multi-year forward projection (2023 → 2050). Wright's Law learning curves reduce costs each period. Shows when the market flips for each resource type.",
    sweep: "Full 270-scenario parametric sweep: 2 conditions × 3 demand × 5 price × 3 PPA × 3 gas friction. Comprehensive but takes longer."
};

// ── Mode switching ──
document.querySelectorAll('.mode-toggle .toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.mode-toggle .toggle-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const mode = btn.dataset.mode;
        document.getElementById('modeDescription').textContent = MODE_DESCRIPTIONS[mode];
        document.getElementById('trajectorySettings').style.display = mode === 'trajectory' ? '' : 'none';
    });
});

// ── ISO selection ──
document.querySelectorAll('.iso-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const mode = document.querySelector('.mode-toggle .toggle-btn.active')?.dataset.mode;
        if (mode === 'snapshot') {
            // Single select in snapshot mode
            document.querySelectorAll('.iso-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        } else {
            // Multi-select in trajectory/sweep mode
            btn.classList.toggle('active');
        }
        updateISOSummary();
        updateGeothermalVisibility();
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

// ── Nuclear retirement slider ──
const slider = document.getElementById('nuclearRetirementSlider');
const sliderValue = document.getElementById('nuclearRetirementValue');
const sliderInput = document.getElementById('nuclear_retirement');

slider.addEventListener('input', () => {
    sliderValue.textContent = `$${slider.value}`;
    sliderInput.value = slider.value;
});

sliderInput.addEventListener('input', () => {
    const val = Math.max(10, Math.min(60, parseInt(sliderInput.value) || 30));
    slider.value = val;
    sliderValue.textContent = `$${val}`;
});

// ── Learning curves toggle ──
document.querySelectorAll('#learningToggle .toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const isOn = btn.dataset.value === 'On';
        document.getElementById('learningSpeedGroup').style.display = isOn ? '' : 'none';
    });
});

// ── Form submission ──
document.getElementById('simulationForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const submitBtn = document.getElementById('submitBtn');
    const submitText = submitBtn.querySelector('.submit-text');
    const submitSpinner = submitBtn.querySelector('.submit-spinner');
    const status = document.getElementById('submitStatus');

    // Collect form data
    const params = collectFormData();

    // UI feedback
    submitBtn.disabled = true;
    submitText.style.display = 'none';
    submitSpinner.style.display = '';
    status.textContent = '';
    status.className = 'submit-status';

    try {
        const mode = document.querySelector('.mode-toggle .toggle-btn.active').dataset.mode;
        const endpoint = mode === 'sweep' ? '/api/sweep' : '/api/simulate';

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

        if (mode === 'sweep') {
            // Poll for sweep completion
            status.textContent = `Sweep started — job ${result.job_id}. Polling for results...`;
            pollSweepStatus(result.job_id);
        } else {
            // Store results and redirect
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
    const selectedISOs = Array.from(document.querySelectorAll('.iso-btn.active')).map(b => b.dataset.iso);

    const params = {
        mode: mode,
        isos: selectedISOs,
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
        incentives: {
            ptc_wind: parseFloat(document.getElementById('ptc_wind').value) || 0,
            ptc_solar: parseFloat(document.getElementById('ptc_solar').value) || 0,
            ptc_nuclear_existing: parseFloat(document.getElementById('ptc_nuclear_existing').value) || 0,
            ptc_nuclear_new: parseFloat(document.getElementById('ptc_nuclear_new').value) || 0,
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
        q45: document.querySelector('#q45Toggle .toggle-btn.active')?.dataset.value === '1',
        demand_growth: document.querySelector('#demandToggle .toggle-btn.active')?.dataset.value || 'Medium',
        ppa_level: document.querySelector('#ppaToggle .toggle-btn.active')?.dataset.value || 'Medium',
        gas_friction: document.querySelector('#gasFrictionToggle .toggle-btn.active')?.dataset.value || 'Medium',
        nuclear_retirement_threshold: parseFloat(document.getElementById('nuclear_retirement').value),
    };

    // Trajectory-specific params
    if (mode === 'trajectory') {
        params.condition = document.querySelector('#conditionToggle .toggle-btn.active')?.dataset.value || 'Facilitating';
        params.years = Array.from(document.querySelectorAll('#trajectorySettings input[type="checkbox"]:checked'))
            .map(cb => parseInt(cb.value));
        params.learning_curves = document.querySelector('#learningToggle .toggle-btn.active')?.dataset.value === 'On';
        params.learning_speed = document.querySelector('#learningSpeedToggle .toggle-btn.active')?.dataset.value || 'Medium';
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

// ── Initialize ──
updateISOSummary();
updateGeothermalVisibility();
