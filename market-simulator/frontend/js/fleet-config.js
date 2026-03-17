/**
 * Fleet Configuration Page — manages plant status overrides for Constellation/Calpine fleet.
 * Stores overrides in localStorage for persistence across page navigation.
 */

const STORAGE_KEY = 'fleet_overrides';
const FOSSIL_FUELS = new Set(['Gas', 'Coal', 'Oil', 'Gas/Oil', 'Oil/Coal']);

let allPlants = [];
let overrides = {};  // { plant_id: 'Operating' | 'Retired' | 'CCS Retrofit' }

// ── Load fleet data ──
async function loadFleetData() {
    try {
        const resp = await fetch('/api/fleet-config');
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        allPlants = await resp.json();
    } catch (err) {
        console.error('Failed to load fleet data:', err);
        allPlants = [];
    }

    // Load saved overrides
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) overrides = JSON.parse(saved);
    } catch (e) {
        overrides = {};
    }

    renderPlantList();
    updateSummary();
}

// ── Render plant list grouped by ISO → Fuel Type ──
function renderPlantList() {
    const container = document.getElementById('plantListContainer');
    const searchTerm = (document.getElementById('searchBar')?.value || '').toLowerCase();

    // Group by ISO
    const byISO = {};
    for (const p of allPlants) {
        if (searchTerm) {
            const haystack = `${p.name} ${p.state} ${p.iso} ${p.fuel_type}`.toLowerCase();
            if (!haystack.includes(searchTerm)) continue;
        }
        if (!byISO[p.iso]) byISO[p.iso] = {};
        if (!byISO[p.iso][p.fuel_type]) byISO[p.iso][p.fuel_type] = [];
        byISO[p.iso][p.fuel_type].push(p);
    }

    // Sort ISOs: US ISOs first, then international
    const US_ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP'];
    const sortedISOs = Object.keys(byISO).sort((a, b) => {
        const aIdx = US_ISOS.indexOf(a);
        const bIdx = US_ISOS.indexOf(b);
        if (aIdx >= 0 && bIdx >= 0) return aIdx - bIdx;
        if (aIdx >= 0) return -1;
        if (bIdx >= 0) return 1;
        return a.localeCompare(b);
    });

    let html = '';
    for (const iso of sortedISOs) {
        const fuelGroups = byISO[iso];
        const isoPlants = Object.values(fuelGroups).flat();
        const isoCapMW = isoPlants.reduce((s, p) => s + p.capacity_mw, 0);

        html += `<div class="fleet-section">`;
        html += `<div class="fleet-section-header" onclick="toggleSection(this)">
            <h3>${iso} <span class="count">(${isoPlants.length} plants · ${(isoCapMW/1000).toFixed(1)} GW)</span></h3>
            <span class="chevron">▼</span>
        </div>`;
        html += `<div class="fleet-section-body">`;

        // Sort fuel types: fossil first (expanded), clean collapsed
        const fuels = Object.keys(fuelGroups).sort((a, b) => {
            const aF = FOSSIL_FUELS.has(a) ? 0 : 1;
            const bF = FOSSIL_FUELS.has(b) ? 0 : 1;
            return aF - bF || a.localeCompare(b);
        });

        for (const fuel of fuels) {
            const plants = fuelGroups[fuel];
            const isFossil = FOSSIL_FUELS.has(fuel);
            const fuelCapMW = plants.reduce((s, p) => s + p.capacity_mw, 0);

            html += `<div style="margin-left: 16px; margin-bottom: 8px;">`;
            html += `<div class="fleet-section-header ${isFossil ? '' : 'collapsed'}" onclick="toggleSection(this)" style="background: ${isFossil ? '#fef2f2' : '#f0fdf4'}; border-color: ${isFossil ? '#fca5a5' : '#86efac'};">
                <h3 style="font-size: 0.9rem;">${fuel} <span class="count">(${plants.length} · ${(fuelCapMW/1000).toFixed(1)} GW)</span></h3>
                <span class="chevron">▼</span>
            </div>`;
            html += `<div class="fleet-section-body ${isFossil ? '' : 'collapsed'}">`;

            // Column headers
            html += `<div class="plant-row" style="font-weight: 600; font-size: 0.8rem; color: var(--text-muted); border-bottom: 2px solid var(--border);">
                <div>Plant Name</div>
                <div>Capacity</div>
                <div class="plant-state">State</div>
                <div class="plant-meta">Year</div>
                <div>Status</div>
            </div>`;

            for (const p of plants.sort((a, b) => b.capacity_mw - a.capacity_mw)) {
                const currentStatus = overrides[p.id] || p.status;
                const isModified = overrides[p.id] && overrides[p.id] !== p.status;
                const ccsOption = p.ccs_eligible ?
                    `<option value="CCS Retrofit" ${currentStatus === 'CCS Retrofit' ? 'selected' : ''}>CCS Retrofit</option>` : '';

                html += `<div class="plant-row">
                    <div class="plant-name">${p.name}</div>
                    <div>${p.capacity_mw.toFixed(0)} MW</div>
                    <div class="plant-state">${p.state || '—'}</div>
                    <div class="plant-meta">${p.year_built || '—'}</div>
                    <div>
                        <select onchange="setOverride('${p.id}', this.value, '${p.status}')" class="${isModified ? 'modified' : ''}">
                            <option value="Operating" ${currentStatus === 'Operating' ? 'selected' : ''}>Operating</option>
                            <option value="Retired" ${currentStatus === 'Retired' ? 'selected' : ''}>Retired</option>
                            ${ccsOption}
                        </select>
                    </div>
                </div>`;
            }

            html += `</div></div>`;  // close fuel section body + wrapper
        }

        html += `</div></div>`;  // close ISO section body + section
    }

    container.innerHTML = html || '<p style="color: var(--text-muted); padding: 20px;">No plants match your search.</p>';
}

// ── Section toggle ──
function toggleSection(header) {
    header.classList.toggle('collapsed');
    const body = header.nextElementSibling;
    if (body) body.classList.toggle('collapsed');
}

// ── Set override ──
function setOverride(plantId, newStatus, defaultStatus) {
    if (newStatus === defaultStatus) {
        delete overrides[plantId];
    } else {
        overrides[plantId] = newStatus;
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(overrides));
    updateSummary();

    // Update select styling
    const selects = document.querySelectorAll(`select`);
    selects.forEach(sel => {
        const row = sel.closest('.plant-row');
        if (row) {
            const modified = sel.value !== sel.querySelector('option').value;
            sel.className = modified ? 'modified' : '';
        }
    });
}

// ── Bulk actions ──
function bulkAction(fuelType, newStatus, onlyCCSEligible = false) {
    for (const p of allPlants) {
        if (p.fuel_type !== fuelType && !p.fuel_type.includes(fuelType)) continue;
        if (onlyCCSEligible && !p.ccs_eligible) continue;
        if (newStatus === p.status) {
            delete overrides[p.id];
        } else {
            overrides[p.id] = newStatus;
        }
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(overrides));
    renderPlantList();
    updateSummary();
}

function resetAll() {
    overrides = {};
    localStorage.setItem(STORAGE_KEY, JSON.stringify(overrides));
    renderPlantList();
    updateSummary();
}

// ── Update summary stats ──
function updateSummary() {
    const statuses = allPlants.map(p => overrides[p.id] || p.status);
    const operating = statuses.filter(s => s === 'Operating').length;
    const retired = statuses.filter(s => s === 'Retired').length;
    const ccs = statuses.filter(s => s === 'CCS Retrofit').length;
    const modified = Object.keys(overrides).length;
    const totalCap = allPlants.reduce((s, p) => {
        const st = overrides[p.id] || p.status;
        return st !== 'Retired' ? s + p.capacity_mw : s;
    }, 0);

    document.getElementById('statTotal').textContent = allPlants.length;
    document.getElementById('statOperating').textContent = operating;
    document.getElementById('statRetired').textContent = retired;
    document.getElementById('statCCS').textContent = ccs;
    document.getElementById('statCapacity').textContent = `${(totalCap/1000).toFixed(1)} GW`;
    document.getElementById('statModified').textContent = modified;
    document.getElementById('changesCount').textContent =
        modified > 0 ? `${modified} plant${modified > 1 ? 's' : ''} modified` : 'No changes';
}

// ── Save and return to setup ──
function saveAndReturn() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(overrides));
    window.location.href = '/setup';
}

// ── Search ──
document.getElementById('searchBar')?.addEventListener('input', () => {
    renderPlantList();
});

// ── Init ──
loadFleetData();
