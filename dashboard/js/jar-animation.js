/**
 * Jar Animation Engine v4 — DOM-based glassmorphism jars with CSS animations
 *
 * Each jar shows what happens to an ISO's GRID under a procurement strategy.
 * At 0% participation: baseline grid mix (solid balls).
 * As participation increases: new procurement adds to baseline.
 *
 * Visual tiers (CSS classes):
 * - .ball--baseline: solid fill (grid baseline clean)
 * - .ball--claimed: transparent fill + saturated ring (existing clean claimed)
 * - .ball--new: translucent fill (new-build procurement)
 * - .ball--curtailed: diagonal stripe + dashed border (curtailed above rim)
 *
 * Glow: .ball--glow with --glow-color (buyer ISO color on source jar balls)
 *
 * Jar background: linear-gradient rising from bottom via --hms-pct CSS var
 * representing hourly matching score.
 */

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

const STRATEGIES = ['1A', '1B', '2A', '2B', '2C', '3A', '3B', '3C', '3D'];
const ISO_LIST = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP'];
const STRATEGY_LABELS = {
    '1A': 'Consequential\n(Grid Avg)',
    '1B': 'Consequential\n(Fossil Avg)',
    '2A': 'Hourly\n(New-Build)',
    '2B': 'Hourly\n(Grid Baseline)',
    '2C': 'Hourly\n(SSS + Tranches)',
    '3A': 'Annual\n(Same-ISO)',
    '3B': 'Annual\n(Cross-Regional)',
    '3C': 'Annual\n(Same, Exist.)',
    '3D': 'Annual\n(Cross, Exist.)',
};

// Annual grid demand per ISO (TWh) from pipeline_config.py
const GRID_DEMANDS = {
    'CAISO': 224, 'ERCOT': 488, 'PJM': 843,
    'NYISO': 152, 'NEISO': 115, 'MISO': 660, 'SPP': 296,
};

// Existing gas capacity per ISO (GW) from pipeline_config.py
const EXISTING_GAS_GW = {
    'CAISO': 37.0, 'ERCOT': 55.0, 'PJM': 75.0,
    'NYISO': 18.0, 'NEISO': 14.0, 'MISO': 68.0, 'SPP': 32.0,
};

// New-build CCGT annualized cost ($/kW-yr) from pipeline_config.py (Lazard v16.0)
const NEW_CCGT_COST_KW_YR = {
    'CAISO': 112, 'ERCOT': 89, 'PJM': 99, 'NYISO': 114, 'NEISO': 105,
    'MISO': 95, 'SPP': 88,
};


function getResourceColor(resource) {
    const map = {
        'solar': '#F59E0B', 'wind': '#22C55E', 'offshore_wind': '#009688',
        'hydro': '#0EA5E9', 'clean_firm': '#6366F1', 'nuclear': '#6366F1',
        'ccs_ccgt': '#64748B', 'ccs': '#64748B', 'geothermal': '#D97706',
        'battery': '#06B6D4', 'battery4': '#06B6D4', 'battery8': '#0891B2',
        'ldes': '#E91E63', 'green_h2': '#10B981',
        'storage': '#EF4444', 'gap': '#D1D5DB',
        'new_vre': '#22C55E', 'new_build_vre': '#22C55E',
        'existing_merchant': '#6366F1', 'existing_vre': '#F59E0B',
    };
    if (typeof RESOURCE_COLORS !== 'undefined') {
        const key = resource.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
        if (RESOURCE_COLORS[key]) return RESOURCE_COLORS[key];
    }
    return map[resource] || '#94A3B8';
}

function getResourceLabel(resource) {
    const labels = {
        'solar': 'Solar', 'wind': 'Wind', 'offshore_wind': 'Offshore Wind',
        'hydro': 'Hydro', 'clean_firm': 'Nuclear', 'nuclear': 'Nuclear',
        'ccs_ccgt': 'CCS-CCGT', 'ccs': 'CCS', 'geothermal': 'Geothermal',
        'battery': 'Battery 4hr', 'battery4': 'Battery 4hr', 'battery8': 'Battery 8hr',
        'ldes': 'LDES', 'green_h2': 'Green H₂', 'storage': 'Storage',
        'sss_allocation': 'SSS Allocation', 'existing_nuclear': 'Nuclear',
        'nuclear_uprate': 'Nuclear Uprate', 'existing_vre': 'Existing VRE',
        'grid_clean': 'Grid Clean', 'existing_vre_hydro': 'Hydro (Existing)',
        'new_vre': 'New VRE', 'new_build_vre': 'New VRE',
        'existing_merchant': 'Merchant Clean',
    };
    return labels[resource] || resource.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function getIsoColor(iso) {
    const map = {
        'CAISO': '#F59E0B', 'ERCOT': '#22C55E', 'PJM': '#0EA5E9',
        'NYISO': '#E91E63', 'NEISO': '#9C27B0', 'MISO': '#F97316', 'SPP': '#14B8A6'
    };
    if (typeof ISO_COLORS !== 'undefined' && ISO_COLORS[iso]) return ISO_COLORS[iso];
    return map[iso] || '#6B7280';
}


// ═══════════════════════════════════════════════════════════════════════════════
// JAR CLASS — DOM-based glassmorphism jar with flexbox ball stacking
// ═══════════════════════════════════════════════════════════════════════════════

class Jar {
    constructor(strategy, iso) {
        this.strategy = strategy;
        this.iso = iso;
        this.data = null;
        this.gridBaseline = null;
        this.ballItems = [];  // data-only ball descriptors
        this.el = null;       // DOM element (created on first render)
    }

    /**
     * Set ball data using absolute 100-ball scale (1 ball = 1% of grid demand).
     *
     * @param {object} gridRecord - Grid-centric aggregated record: {n: {res: twh}, buyerFlows: {iso: {res: twh}}}
     * @param {object} gridBaseline - Grid baseline clean shares {res: pct, totalPct: N}
     * @param {boolean} showGlow - Whether to show cross-ISO buyer glow
     * @param {number} curtTwh - Curtailment TWh for this grid ISO under this strategy
     */
    setBalls(gridRecord, gridBaseline, showGlow, curtTwh) {
        this.data = gridRecord;
        this.gridBaseline = gridBaseline;

        const iso = this.iso;
        const bl = gridBaseline || {};
        const gridDemand = GRID_DEMANDS[iso] || 300;

        const items = [];

        // 1. Grid baseline balls — each ball = 1% of grid demand (saturated/solid)
        for (const [res, pct] of Object.entries(bl)) {
            if (res === 'totalPct' || pct <= 0) continue;
            items.push({
                resource: res,
                count: Math.max(1, Math.round(pct)),  // 1 ball per % of demand
                tier: 'baseline',
                glowIso: null,
            });
        }

        // 2. New-build procurement balls — transparent/outline
        //    gridRecord.n has {resource: twh} aggregated from all buyers targeting this grid
        if (gridRecord && gridRecord.n) {
            // Build per-resource ball counts, tracking buyer glow per ball
            const buyerFlows = gridRecord.buyerFlows || {};

            for (const [res, twh] of Object.entries(gridRecord.n)) {
                if (twh <= 0) continue;
                const pctOfDemand = twh / gridDemand * 100;
                const ballCount = Math.max(1, Math.round(pctOfDemand));

                if (showGlow && Object.keys(buyerFlows).length > 0) {
                    // Distribute balls across buyer ISOs proportionally
                    let totalBuyerTwh = 0;
                    const buyerTwhForRes = {};
                    for (const [buyerIso, resources] of Object.entries(buyerFlows)) {
                        const bt = (resources[res] || 0);
                        if (bt > 0) {
                            buyerTwhForRes[buyerIso] = bt;
                            totalBuyerTwh += bt;
                        }
                    }

                    if (totalBuyerTwh > 0) {
                        let allocated = 0;
                        const entries = Object.entries(buyerTwhForRes);
                        for (let i = 0; i < entries.length; i++) {
                            const [buyerIso, bt] = entries[i];
                            const share = i < entries.length - 1
                                ? Math.max(1, Math.round(bt / totalBuyerTwh * ballCount))
                                : Math.max(1, ballCount - allocated);
                            items.push({ resource: res, count: share, tier: 'new', glowIso: buyerIso });
                            allocated += share;
                        }
                    } else {
                        items.push({ resource: res, count: ballCount, tier: 'new', glowIso: null });
                    }
                } else {
                    items.push({ resource: res, count: ballCount, tier: 'new', glowIso: null });
                }
            }
        }

        // 3. Curtailment balls — above the rim (1 ball per % of demand)
        const curtItems = [];
        const effectiveCurt = curtTwh || (gridRecord && gridRecord.curtTwh) || 0;
        if (effectiveCurt > 0) {
            const curtPct = effectiveCurt / gridDemand * 100;
            const curtBalls = Math.max(1, Math.min(30, Math.round(curtPct)));  // cap at 30 for readability
            const vreResources = ['solar', 'wind'];
            for (const res of vreResources) {
                curtItems.push({
                    resource: res,
                    count: Math.max(1, Math.round(curtBalls / vreResources.length)),
                    tier: 'curtailed',
                    glowIso: null,
                });
            }
        }

        // Sort non-curtailed: baseline bottom, new top
        const tierOrder = { 'baseline': 0, 'new': 1 };
        items.sort((a, b) => {
            const ta = tierOrder[a.tier] || 0;
            const tb_order = tierOrder[b.tier] || 0;
            if (ta !== tb_order) return ta - tb_order;
            return (b.count || 0) - (a.count || 0);
        });

        // Flatten to individual ball descriptors
        this.ballItems = [];
        const allItems = [...items, ...curtItems];
        for (const item of allItems) {
            if (!item.count) continue;
            for (let i = 0; i < item.count; i++) {
                this.ballItems.push({
                    resource: item.resource,
                    tier: item.tier,
                    glowIso: item.glowIso,
                    color: getResourceColor(item.resource),
                    glowColor: item.glowIso ? getIsoColor(item.glowIso) : null,
                });
            }
        }
    }

    _mapClaimedResource(res) {
        const map = {
            'grid_clean': 'clean_firm',
            'sss_allocation': 'clean_firm',
            'existing_nuclear': 'clean_firm',
            'nuclear_uprate': 'clean_firm',
            'existing_vre': 'solar',
            'existing_vre_hydro': 'hydro',
            'ccs': 'ccs_ccgt',
            'ccs_ccgt': 'ccs_ccgt',
        };
        return map[res] || res;
    }

    /**
     * Render/update the DOM element for this jar.
     * @param {number} ballSize - ball diameter in px
     */
    renderDOM(ballSize) {
        if (!this.el) {
            this.el = document.createElement('div');
            this.el.className = 'jar-dom';
            this.el.dataset.strategy = this.strategy;
            this.el.dataset.iso = this.iso;
        }

        // Set HMS gradient height — use gridCleanPct (grid-centric) or fallback
        const hms = this.data && this.data.hms;
        const gridCleanPct = this.data && this.data.gridCleanPct;
        const blPctFallback = this.gridBaseline ? (this.gridBaseline.totalPct || 0) : 0;
        const hmsPct = hms != null ? Math.min(100, Math.max(0, hms)) :
                       (gridCleanPct != null ? Math.min(100, Math.max(0, gridCleanPct)) :
                       Math.min(100, Math.max(0, blPctFallback)));
        this.el.style.setProperty('--hms-pct', hmsPct + '%');
        this.el.style.setProperty('--ball-size', ballSize + 'px');

        // Separate normal and curtailed balls
        const normalBalls = this.ballItems.filter(b => b.tier !== 'curtailed');
        const curtailedBalls = this.ballItems.filter(b => b.tier === 'curtailed');

        // Build inner HTML in one batch for performance
        let html = '';

        // Normal balls (flexbox wrap-reverse stacks from bottom)
        let ballIndex = 0;
        for (const b of normalBalls) {
            const delay = ((ballIndex * 0.13) % 3).toFixed(2);
            const tierClass = `ball ball--${b.tier}`;
            const glowClass = b.glowIso ? ' ball--glow' : '';
            const glowStyle = b.glowColor ? `;--glow-color:${b.glowColor}` : '';
            html += `<div class="${tierClass}${glowClass}" style="--ball-color:${b.color};--float-delay:${delay}s${glowStyle}" data-resource="${b.resource}" data-tier="${b.tier}"${b.glowIso ? ` data-glow-iso="${b.glowIso}"` : ''}></div>`;
            ballIndex++;
        }

        // Curtailed zone (above rim)
        if (curtailedBalls.length > 0) {
            html += '<div class="jar-curtailed-zone">';
            for (const b of curtailedBalls) {
                const delay = ((ballIndex * 0.13) % 3).toFixed(2);
                html += `<div class="ball ball--curtailed" style="--ball-color:${b.color};--float-delay:${delay}s" data-resource="${b.resource}" data-tier="curtailed"></div>`;
                ballIndex++;
            }
            html += '</div>';
        }

        // HMS label
        const displayPct = hms != null ? hms : (gridCleanPct != null ? gridCleanPct : null);
        if (displayPct != null && displayPct >= 0.5) {
            const label = hms != null ? `${Math.round(displayPct)}% Grid HMS` : `${Math.round(displayPct)}%`;
            html += `<div class="jar-hms-label">${label}</div>`;
        } else if (this.gridBaseline) {
            const blPct = this.gridBaseline.totalPct || 0;
            if (blPct > 0) {
                html += `<div class="jar-hms-label" style="opacity:0.45">${Math.round(blPct)}%</div>`;
            }
        }

        this.el.innerHTML = html;
        return this.el;
    }

    _getTotalCleanTwh() {
        if (!this.data) return 0;
        let total = 0;
        if (this.data.e) {
            for (const t of Object.values(this.data.e)) {
                if (t > 0) total += t;
            }
        }
        if (this.data.n) {
            for (const t of Object.values(this.data.n)) {
                if (t > 0) total += t;
            }
        }
        return total;
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// JAR GRID — Canvas grid + DOM jar overlay controller
// ═══════════════════════════════════════════════════════════════════════════════

class JarGrid {
    constructor(canvasId, tooltipId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.tooltipEl = document.getElementById(tooltipId);
        this.overlayEl = document.getElementById('jarOverlay');

        this.participation = 10;
        this.threshold = 90;
        this.data = null;
        this.gridBaseline = null;
        this.jars = [];

        // Active strategies (user-selectable via checkboxes)
        this.activeStrategies = ['1B', '2C', '3D'];

        // Layout
        this.rowHeaderWidth = 0;
        this.colHeaderHeight = 0;
        this.jarWidth = 0;
        this.jarHeight = 0;

        // Interaction
        this.hoveredJar = null;
        this.showCrossIsoGlow = false;

        this._boundResize = this._onResize.bind(this);
        window.addEventListener('resize', this._boundResize);

        this._onResize();
    }

    init(deploymentData) {
        this.data = deploymentData;
        this.gridBaseline = deploymentData.gridBaseline || {};
        this._buildJars();
        this._updateData();
        this._drawCanvas();
    }

    setParticipation(pct) {
        if (this.participation === pct) return;
        this.participation = pct;
        this._updateData();
        this._drawCanvas();
    }

    setThreshold(thr) {
        if (this.threshold === thr) return;
        this.threshold = thr;
        this._updateData();
        this._drawCanvas();
    }

    setActiveStrategies(strategies) {
        this.activeStrategies = strategies;
        this._onResize();
    }

    _onResize() {
        const dpr = window.devicePixelRatio || 1;
        const container = this.canvas.parentElement;
        const rect = container.getBoundingClientRect();

        const isMobile = window.innerWidth < 768;
        const isTablet = window.innerWidth < 1024;

        const numCols = this.activeStrategies.length;

        // Row headers = ISO names (shorter), col headers = strategy IDs + gas GW sub-label
        this.rowHeaderWidth = isMobile ? 70 : (isTablet ? 90 : 115);
        this.colHeaderHeight = isMobile ? 110 : 140;

        const availWidth = rect.width - this.rowHeaderWidth;
        const jarW = Math.max(55, Math.floor(availWidth / numCols));
        const jarH = isMobile ? 150 : (isTablet ? 190 : 240);
        // Extra vertical room: curtailment overflows above (up to 30 balls × ballSize), HMS label below
        const curtailPad = isMobile ? 50 : 80;
        const hmsLabelPad = 18;
        const rowGap = (isMobile ? 14 : (isTablet ? 20 : 26)) + curtailPad + hmsLabelPad;

        this.jarWidth = jarW;
        this.jarHeight = jarH;
        this.rowGap = rowGap;
        this.curtailPad = curtailPad;
        this.rowStride = jarH + rowGap;

        const totalWidth = this.rowHeaderWidth + jarW * numCols;
        const totalHeight = this.colHeaderHeight + this.rowStride * ISO_LIST.length;

        this.canvas.style.width = totalWidth + 'px';
        this.canvas.style.height = totalHeight + 'px';
        this.canvas.width = totalWidth * dpr;
        this.canvas.height = totalHeight * dpr;
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        // Size overlay to match canvas
        if (this.overlayEl) {
            this.overlayEl.style.width = totalWidth + 'px';
            this.overlayEl.style.height = totalHeight + 'px';
        }

        this._buildJars();
        this._updateData();
        this._drawCanvas();
    }

    _buildJars() {
        // Remove old DOM elements
        if (this.overlayEl) this.overlayEl.innerHTML = '';
        this._tooltipWired = false;

        // Rows = ISOs (7), Cols = active strategies (variable)
        this.jars = [];
        for (let row = 0; row < ISO_LIST.length; row++) {
            for (let col = 0; col < this.activeStrategies.length; col++) {
                const jar = new Jar(this.activeStrategies[col], ISO_LIST[row]);
                this.jars.push(jar);
            }
        }
    }

    _findClosestKey(available, target) {
        let best = null, bestDist = Infinity;
        for (const key of available) {
            const dist = Math.abs(parseFloat(key) - target);
            if (dist < bestDist) { bestDist = dist; best = key; }
        }
        return best;
    }

    _buildCrossIsoFlows(strategyId) {
        if (!this.data || !this.data.data[strategyId]) return {};
        const flows = {};
        const stratData = this.data.data[strategyId];

        for (const buyerIso of ISO_LIST) {
            const isoData = stratData[buyerIso];
            if (!isoData) continue;
            const pk = this._findClosestKey(Object.keys(isoData), this.participation);
            if (!pk) continue;
            const tk = this._findClosestKey(Object.keys(isoData[pk]), this.threshold);
            if (!tk) continue;
            const record = isoData[pk][tk];
            if (!record || !record.x) continue;

            for (const [srcIso, resources] of Object.entries(record.x)) {
                if (!flows[srcIso]) flows[srcIso] = {};
                if (!flows[srcIso][buyerIso]) flows[srcIso][buyerIso] = {};
                for (const [res, twh] of Object.entries(resources)) {
                    if (twh > 0) {
                        flows[srcIso][buyerIso][res] = (flows[srcIso][buyerIso][res] || 0) + twh;
                    }
                }
            }
        }
        return flows;
    }

    _updateData() {
        if (!this.data) return;

        // Auto-size balls so 100 balls fills the jar exactly
        const isMobile = window.innerWidth < 768;
        const jarPadding = 7;  // 3px CSS padding + 4px inset
        const gapPx = 2;
        const jarInnerW = this.jarWidth - 8 - jarPadding * 2;
        const jarInnerH = this.jarHeight - 8 - jarPadding * 2;

        // Pick balls-per-row, then compute ball size to fill height with ceil(100/bpr) rows
        const ballsPerRow = isMobile ? 7 : 10;
        const rowsNeeded = Math.ceil(100 / ballsPerRow);  // 10 desktop, 15 mobile
        const ballSize = Math.max(4, Math.floor((jarInnerH - gapPx * (rowsNeeded - 1)) / rowsNeeded));

        // ── Grid-centric aggregation ──
        // For each strategy, aggregate all buyer records into grid-centric view:
        // gridView[strat][gridIso] = {n: {res: totalTwh}, buyerFlows: {buyerIso: {res: twh}}, hms, curtTwh, ...}
        const gridViews = {};

        for (const strat of this.activeStrategies) {
            const stratData = this.data.data[strat];
            if (!stratData) { gridViews[strat] = {}; continue; }

            const view = {};
            // Initialize each grid ISO
            for (const iso of ISO_LIST) {
                view[iso] = { n: {}, buyerFlows: {}, curtTwh: 0, hms: null, gridCleanPct: null, gasGw: null };
            }

            // Walk all buyer ISOs and redistribute resources to destination grids
            for (const buyerIso of ISO_LIST) {
                const isoData = stratData[buyerIso];
                if (!isoData) continue;
                const pk = this._findClosestKey(Object.keys(isoData), this.participation);
                if (!pk) continue;
                const tk = this._findClosestKey(Object.keys(isoData[pk]), this.threshold);
                if (!tk) continue;
                const record = isoData[pk][tk];
                if (!record) continue;

                // Same-ISO new-build → goes to buyer's grid
                if (record.n) {
                    for (const [res, twh] of Object.entries(record.n)) {
                        if (twh > 0) {
                            view[buyerIso].n[res] = (view[buyerIso].n[res] || 0) + twh;
                        }
                    }
                }

                // Existing claims (e) — already in grid baseline, no new balls needed.
                // But store for detail panel / metrics
                if (record.e) {
                    if (!view[buyerIso].e) view[buyerIso].e = {};
                    for (const [res, twh] of Object.entries(record.e)) {
                        if (twh > 0) {
                            view[buyerIso].e[res] = (view[buyerIso].e[res] || 0) + twh;
                        }
                    }
                }

                // Cross-ISO flows → go to SOURCE grid, with buyer glow tracking
                if (record.x) {
                    for (const [srcIso, resources] of Object.entries(record.x)) {
                        if (!view[srcIso]) continue;  // unknown ISO
                        for (const [res, twh] of Object.entries(resources)) {
                            if (twh > 0) {
                                view[srcIso].n[res] = (view[srcIso].n[res] || 0) + twh;
                                // Track buyer origin for glow
                                if (!view[srcIso].buyerFlows[buyerIso]) {
                                    view[srcIso].buyerFlows[buyerIso] = {};
                                }
                                view[srcIso].buyerFlows[buyerIso][res] =
                                    (view[srcIso].buyerFlows[buyerIso][res] || 0) + twh;
                            }
                        }
                    }
                }

                // Take dispatch metrics from the buyer record
                // Prefer grid-centric fields (gridHms, gridCurtTwh, gridGasGw) when available
                // Grid-centric curtailment only
                if (record.gridCurtTwh > 0) {
                    view[buyerIso].curtTwh = record.gridCurtTwh;
                }
                // Grid-centric metrics ONLY (no buyer-centric fallback)
                if (record.gridHms != null) {
                    view[buyerIso].hms = record.gridHms;
                }
                if (record.gridAggCleanPct != null) {
                    view[buyerIso].gridCleanPct = record.gridAggCleanPct;
                }
                if (record.gridGasGw != null) {
                    view[buyerIso].gasGw = record.gridGasGw;
                } else if (record.gasGw != null && view[buyerIso].gasGw == null) {
                    // Fallback to buyer-centric gasGw when grid-centric not available
                    view[buyerIso].gasGw = record.gasGw;
                }
                // Preserve cost/co2 data
                if (record.tc != null) view[buyerIso].tc = (view[buyerIso].tc || 0) + record.tc;
                if (record.co2 != null) view[buyerIso].co2 = (view[buyerIso].co2 || 0) + record.co2;
                if (record.bl != null) view[buyerIso].bl = (view[buyerIso].bl || 0) + record.bl;
            }

            gridViews[strat] = view;
        }

        for (const jar of this.jars) {
            const gridView = gridViews[jar.strategy] || {};
            const gridRecord = gridView[jar.iso] || null;
            const bl = this.gridBaseline[jar.iso] || null;
            const curtTwh = gridRecord ? gridRecord.curtTwh : 0;

            jar.setBalls(gridRecord, bl, this.showCrossIsoGlow, curtTwh);
            jar.renderDOM(ballSize);
        }

        this._positionDOMJars();
        this._wireTooltips();

        // Compute per-strategy metrics (gas GW, curtailed TWh, total cost) from grid views
        this.strategyGasGw = {};
        this.strategyNewGasGw = {};
        this.strategyCurtTwh = {};
        this.strategyCostM = {};
        this.strategyGasCostM = {};
        // Sum of existing gas across all ISOs (baseline without any clean procurement)
        const totalExistingGas = ISO_LIST.reduce((s, iso) => s + (EXISTING_GAS_GW[iso] || 0), 0);

        for (const strat of this.activeStrategies) {
            let totalGasRemaining = 0, totalNewGas = 0, totalCurt = 0, totalCost = 0, totalGasCost = 0;
            const view = gridViews[strat] || {};
            for (const iso of ISO_LIST) {
                const gv = view[iso];
                if (!gv) continue;
                // gasGw = total gas backup needed on this grid (decreases as clean increases)
                const gasGw = gv.gasGw != null ? gv.gasGw : (EXISTING_GAS_GW[iso] || 0);
                totalGasRemaining += gasGw;
                const newGas = Math.max(0, gasGw - (EXISTING_GAS_GW[iso] || 0));
                totalNewGas += newGas;
                totalGasCost += newGas * (NEW_CCGT_COST_KW_YR[iso] || 100) * 1000;
                totalCurt += gv.curtTwh || 0;
                totalCost += gv.tc || 0;
            }
            // Primary: gas displaced (positive = good)
            this.strategyGasGw[strat] = Math.max(0, totalExistingGas - totalGasRemaining);
            // Secondary: new gas needed beyond existing + cost
            this.strategyNewGasGw[strat] = totalNewGas;
            this.strategyCurtTwh[strat] = totalCurt;
            this.strategyGasCostM[strat] = totalGasCost;
            this.strategyCostM[strat] = totalCost + totalGasCost;
        }

        // Fire stats callback
        if (this.onStatsUpdate) {
            let totalPaidTwh = 0, totalCO2 = 0;
            for (const jar of this.jars) {
                if (!jar.data) continue;
                totalPaidTwh += jar._getTotalCleanTwh();
                totalCO2 += jar.data.co2 || 0;
            }
            this.onStatsUpdate({
                totalPaidTwh,
                totalCO2Mt: totalCO2,
                totalRealCO2Mt: totalCO2,
            });
        }
    }

    _positionDOMJars() {
        if (!this.overlayEl) return;

        const insetX = 4;
        const insetY = 4;
        const jarContentH = this.jarHeight - insetY * 2;
        const jarContentW = this.jarWidth - insetX * 2;
        const numCols = this.activeStrategies.length;

        for (let i = 0; i < this.jars.length; i++) {
            const jar = this.jars[i];
            // Rows = ISOs, Cols = strategies
            const row = Math.floor(i / numCols);
            const col = i % numCols;

            const x = this.rowHeaderWidth + col * this.jarWidth + insetX;
            // Offset jar down by curtailPad so curtailed balls have room above
            const y = this.colHeaderHeight + row * this.rowStride + (this.curtailPad || 0) + insetY;

            const el = jar.el;
            el.style.left = x + 'px';
            el.style.top = y + 'px';
            el.style.width = jarContentW + 'px';
            el.style.height = jarContentH + 'px';

            if (!el.parentElement) {
                this.overlayEl.appendChild(el);
            }
        }
    }

    _wireTooltips() {
        // Event delegation on overlay
        if (this._tooltipWired) return;
        this._tooltipWired = true;

        const self = this;
        const isMobile = window.innerWidth < 768;

        // Desktop: lightweight 1-line hover tooltip
        if (!isMobile) {
            this.overlayEl.addEventListener('mousemove', function(e) {
                const jarEl = e.target.closest('.jar-dom');
                if (!jarEl) {
                    self._hideTooltip();
                    self.hoveredJar = null;
                    return;
                }
                const jar = self._findJarByEl(jarEl);
                if (!jar) return;
                self.hoveredJar = jar;
                self._showMiniTooltip(e.clientX, e.clientY, jar);
            }, true);

            this.overlayEl.addEventListener('mouseleave', function() {
                self._hideTooltip();
                self.hoveredJar = null;
            });
        }

        // Click → detail panel (desktop + mobile)
        this.overlayEl.addEventListener('click', function(e) {
            const jarEl = e.target.closest('.jar-dom');
            if (!jarEl) return;
            const jar = self._findJarByEl(jarEl);
            if (jar) {
                self._hideTooltip();
                self._showDetailPanel(jar);
            }
        });

        // Touch → detail panel (mobile)
        this.overlayEl.addEventListener('touchstart', function(e) {
            const touch = e.touches[0];
            const jarEl = document.elementFromPoint(touch.clientX, touch.clientY);
            const jarDom = jarEl && jarEl.closest('.jar-dom');
            if (jarDom) {
                const jar = self._findJarByEl(jarDom);
                if (jar) {
                    e.preventDefault();
                    self._showDetailPanel(jar);
                }
            }
        }, { passive: false });
    }

    _findJarByEl(el) {
        const strat = el.dataset.strategy;
        const iso = el.dataset.iso;
        return this.jars.find(j => j.strategy === strat && j.iso === iso);
    }

    _drawCanvas() {
        const ctx = this.ctx;
        const w = this.canvas.width / (window.devicePixelRatio || 1);
        const h = this.canvas.height / (window.devicePixelRatio || 1);
        const numCols = this.activeStrategies.length;

        ctx.clearRect(0, 0, w, h);

        // Column headers (strategies — multi-line labels)
        ctx.save();
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const fontSize = this.jarWidth < 70 ? 11 : (this.jarWidth < 100 ? 14 : 16);
        ctx.font = `600 ${fontSize}px 'Plus Jakarta Sans', sans-serif`;
        ctx.fillStyle = '#334155';

        for (let col = 0; col < numCols; col++) {
            const strat = this.activeStrategies[col];
            const x = this.rowHeaderWidth + col * this.jarWidth + this.jarWidth / 2;
            const lines = (STRATEGY_LABELS[strat] || strat).split('\n');
            const labelCenterY = this.colHeaderHeight / 2 - 6;
            for (let li = 0; li < lines.length; li++) {
                ctx.fillText(lines[li], x, labelCenterY + (li - (lines.length - 1) / 2) * (fontSize + 2));
            }
        }
        ctx.restore();

        // Per-strategy metric sub-labels (gas GW, gas cost, curtailed TWh, total cost)
        ctx.save();
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const metricFS = this.jarWidth < 70 ? 9 : (this.jarWidth < 100 ? 11 : 13);
        const metricLineH = metricFS + 3;
        const metricBaseY = this.colHeaderHeight - 3 * metricLineH + metricLineH / 2;

        for (let col = 0; col < numCols; col++) {
            const strat = this.activeStrategies[col];
            const x = this.rowHeaderWidth + col * this.jarWidth + this.jarWidth / 2;

            // Line 1: New gas GW — red if >0, gray if 0
            const newGasGw = this.strategyNewGasGw?.[strat] ?? 0;
            ctx.font = `700 ${metricFS}px 'DM Sans', sans-serif`;
            ctx.fillStyle = newGasGw > 0 ? '#EF4444' : '#9CA3AF';
            ctx.fillText(newGasGw > 0 ? `+${Math.round(newGasGw)} GW gas` : '0 GW gas', x, metricBaseY);

            // Line 2: Curtailed TWh — amber
            const curtTwh = this.strategyCurtTwh?.[strat] ?? 0;
            ctx.font = `600 ${metricFS}px 'DM Sans', sans-serif`;
            ctx.fillStyle = curtTwh > 1 ? '#F59E0B' : '#9CA3AF';
            const curtLabel = curtTwh >= 1000 ? `${(curtTwh / 1000).toFixed(1)}k TWh curt.` :
                              curtTwh >= 1 ? `${Math.round(curtTwh)} TWh curt.` : '0 TWh curt.';
            ctx.fillText(curtLabel, x, metricBaseY + metricLineH);

            // Line 3: Total system cost (clean + gas) — navy
            const costM = this.strategyCostM?.[strat] ?? 0;
            ctx.font = `600 ${metricFS}px 'DM Sans', sans-serif`;
            ctx.fillStyle = '#334155';
            const costLabel = costM >= 1000 ? `$${(costM / 1000).toFixed(1)}B total` :
                              costM > 0 ? `$${Math.round(costM)}M total` : '$0';
            ctx.fillText(costLabel, x, metricBaseY + 2 * metricLineH);
        }
        ctx.restore();

        // Row headers (ISOs)
        ctx.save();
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        const rowFS = this.rowHeaderWidth < 75 ? 13 : (this.rowHeaderWidth < 95 ? 16 : 20);
        ctx.font = `600 ${rowFS}px 'Space Grotesk', sans-serif`;

        for (let row = 0; row < ISO_LIST.length; row++) {
            const iso = ISO_LIST[row];
            // Match the jar's offset so labels center on the actual jar
            const y = this.colHeaderHeight + row * this.rowStride + (this.curtailPad || 0) + this.jarHeight / 2;
            ctx.fillStyle = getIsoColor(iso);
            ctx.fillText(iso, this.rowHeaderWidth - 6, y);
        }
        ctx.restore();

        // Faint grid lines
        ctx.save();
        ctx.strokeStyle = '#E2E8F0';
        ctx.lineWidth = 0.5;
        for (let row = 0; row <= ISO_LIST.length; row++) {
            const y = this.colHeaderHeight + row * this.rowStride;
            ctx.beginPath(); ctx.moveTo(this.rowHeaderWidth, y); ctx.lineTo(w, y); ctx.stroke();
        }
        for (let col = 0; col <= numCols; col++) {
            const x = this.rowHeaderWidth + col * this.jarWidth;
            ctx.beginPath(); ctx.moveTo(x, this.colHeaderHeight); ctx.lineTo(x, h); ctx.stroke();
        }
        ctx.restore();
    }

    _showMiniTooltip(px, py, jar) {
        if (!this.tooltipEl) return;
        const stratLabel = (STRATEGY_LABELS[jar.strategy] || jar.strategy).replace('\n', ' ');
        const hms = jar.data && jar.data.hms != null ? jar.data.hms : (jar.data && jar.data.gridCleanPct != null ? jar.data.gridCleanPct : null);
        const hmsText = hms != null ? ` · ${Math.round(hms)}% Grid HMS` : '';
        this.tooltipEl.innerHTML = `<span style="color:${getIsoColor(jar.iso)};font-weight:700">${jar.iso}</span> — ${stratLabel}${hmsText} <span style="color:var(--text-muted);font-size:0.7rem">(click for details)</span>`;
        this.tooltipEl.style.display = 'block';

        const ttRect = this.tooltipEl.getBoundingClientRect();
        let left = px + 12;
        let top = py - 10;
        if (left + ttRect.width > window.innerWidth - 10) left = px - ttRect.width - 12;
        if (top + ttRect.height > window.innerHeight - 10) top = window.innerHeight - ttRect.height - 10;
        this.tooltipEl.style.left = left + 'px';
        this.tooltipEl.style.top = top + 'px';
    }

    _showDetailPanel(jar) {
        const panel = document.getElementById('jarDetailPanel');
        if (!panel) return;

        let html = `<div class="close-btn" id="detailPanelClose">&times;</div>`;
        html += `<div class="detail-panel-header">
            <strong>${(STRATEGY_LABELS[jar.strategy] || jar.strategy).replace('\n', ' ')}</strong> —
            <span style="color:${getIsoColor(jar.iso)};font-weight:700">${jar.iso}</span>
        </div>`;

        const bl = this.gridBaseline[jar.iso];
        if (bl) {
            html += `<div class="detail-panel-stats">Grid baseline: ${bl.totalPct.toFixed(0)}% clean</div>`;
        }

        if (jar.data) {
            const data = jar.data;

            html += '<div class="detail-panel-stats">';
            if (data.hms != null) html += `<span>Grid Hourly Match: ${Math.round(data.hms)}%</span>`;
            if (data.gridCleanPct != null) html += `<span>Grid Clean: ${Math.round(data.gridCleanPct)}%</span>`;
            if (data.gasGw != null) html += `<span>Gas Backup: ${Math.round(data.gasGw)} GW</span>`;
            if (data.curtTwh != null && data.curtTwh > 0) html += `<span>Curtailed: ${Math.round(data.curtTwh)} TWh</span>`;
            html += '</div>';

            // Resource table
            html += '<table class="detail-panel-table"><thead><tr><th></th><th>Resource</th><th>TWh</th><th>Type</th></tr></thead><tbody>';

            if (data.e) {
                for (const [res, twh] of Object.entries(data.e)) {
                    if (twh <= 0) continue;
                    html += `<tr>
                        <td><span class="tooltip-dot" style="background:${getResourceColor(res)};opacity:0.4;border:2px solid ${getResourceColor(res)}"></span></td>
                        <td>${getResourceLabel(res)}</td>
                        <td class="detail-twh">${twh.toFixed(1)}</td>
                        <td class="detail-type">Existing</td>
                    </tr>`;
                }
            }
            if (data.n) {
                for (const [res, twh] of Object.entries(data.n)) {
                    if (twh <= 0) continue;
                    html += `<tr>
                        <td><span class="tooltip-dot-hollow" style="border-color:${getResourceColor(res)}"></span></td>
                        <td>${getResourceLabel(res)}</td>
                        <td class="detail-twh">${twh.toFixed(1)}</td>
                        <td class="detail-type">New Build</td>
                    </tr>`;
                }
            }
            // Cross-ISO buyer flows (shows which buyers funded new-build on this grid)
            if (data.buyerFlows && Object.keys(data.buyerFlows).length > 0) {
                for (const [buyerIso, resources] of Object.entries(data.buyerFlows)) {
                    for (const [res, twh] of Object.entries(resources)) {
                        if (twh <= 0) continue;
                        html += `<tr>
                            <td><span class="tooltip-iso-badge" style="background:${getIsoColor(buyerIso)}20;color:${getIsoColor(buyerIso)}">${buyerIso}</span></td>
                            <td>${getResourceLabel(res)}</td>
                            <td class="detail-twh">${twh.toFixed(1)}</td>
                            <td class="detail-type">Funded by ${buyerIso}</td>
                        </tr>`;
                    }
                }
            }

            html += '</tbody></table>';

            if (data.co2 > 0) {
                html += `<div class="detail-panel-footer">CO₂ reduced: <strong>${data.co2.toFixed(1)} MtCO₂</strong>${data.co2r ? ` (${data.co2r}% of baseline)` : ''}</div>`;
            }
        }

        panel.innerHTML = html;
        // Force reflow then open
        panel.classList.remove('open');
        requestAnimationFrame(() => {
            panel.style.display = 'block';
            requestAnimationFrame(() => panel.classList.add('open'));
        });

        // Close handlers
        const closeBtn = document.getElementById('detailPanelClose');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this._hideDetailPanel());
        }

        // Click outside to close
        const self = this;
        setTimeout(() => {
            const handler = function(e) {
                if (!panel.contains(e.target) && !e.target.closest('.jar-dom')) {
                    self._hideDetailPanel();
                    document.removeEventListener('click', handler);
                    document.removeEventListener('touchstart', handler);
                }
            };
            document.addEventListener('click', handler);
            document.addEventListener('touchstart', handler, { passive: true });
        }, 100);
    }

    _hideDetailPanel() {
        const panel = document.getElementById('jarDetailPanel');
        if (!panel) return;
        panel.classList.remove('open');
        setTimeout(() => { panel.style.display = 'none'; }, 300);
    }

    _hideTooltip() {
        if (this.tooltipEl) this.tooltipEl.style.display = 'none';
    }

    getAggregateStats() {
        const stats = {};
        for (const strat of this.activeStrategies) {
            let totalNew = 0, totalCross = 0, totalExisting = 0;
            let totalCO2 = 0, totalBaseline = 0;
            const resourceBreakdown = { existing: {}, new: {} };

            for (const jar of this.jars) {
                if (jar.strategy !== strat || !jar.data) continue;
                const d = jar.data;

                if (d.e) {
                    for (const [r, t] of Object.entries(d.e)) {
                        if (t <= 0) continue;
                        totalExisting += t;
                        resourceBreakdown.existing[r] = (resourceBreakdown.existing[r] || 0) + t;
                    }
                }
                if (d.n) {
                    for (const [r, t] of Object.entries(d.n)) {
                        if (t <= 0) continue;
                        totalNew += t;
                        resourceBreakdown.new[r] = (resourceBreakdown.new[r] || 0) + t;
                    }
                }
                // Cross-ISO flows tracked via buyerFlows
                if (d.buyerFlows) {
                    for (const [buyerIso, resources] of Object.entries(d.buyerFlows)) {
                        for (const [r, t] of Object.entries(resources)) {
                            if (t > 0) totalCross += t;
                            // Already counted in d.n above
                        }
                    }
                }
                totalCO2 += d.co2 || 0;
                totalBaseline += d.bl || 0;
            }

            const totalPaid = totalExisting + totalNew;  // cross is already in new

            stats[strat] = {
                existingTwh: totalExisting,
                newTwh: totalNew,
                crossTwh: totalCross,
                totalTwh: totalPaid,
                totalCO2Mt: totalCO2,
                totalRealCO2Mt: totalCO2,
                totalBaselineMt: totalBaseline,
                co2ReductionPct: totalBaseline > 0 ? (totalCO2 / totalBaseline * 100) : 0,
                totalNewGasGw: this.strategyGasGw?.[strat] ?? 0,
                totalCurtTwh: this.strategyCurtTwh?.[strat] ?? 0,
                gasCostM: this.strategyGasCostM?.[strat] ?? 0,
                totalCostM: this.strategyCostM?.[strat] ?? 0,
                resourceBreakdown,
            };
        }
        return stats;
    }

    dispose() {
        window.removeEventListener('resize', this._boundResize);
        if (this.overlayEl) this.overlayEl.innerHTML = '';
    }
}
