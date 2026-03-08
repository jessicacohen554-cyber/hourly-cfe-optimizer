/**
 * Jar Animation Engine — Canvas-based procurement deployment visualization
 *
 * Renders a 7×7 grid of "jars" (ISOs × strategies) filled with colored balls
 * representing resource deployment. Ball saturation indicates existing vs new.
 * Cross-ISO source jars glow when active.
 */

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

const STRATEGIES = ['1A', '1B', '2A', '2B', '2C', '3A', '3B'];
const ISO_LIST = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP'];
const STRATEGY_LABELS = {
    '1A': 'Consequential\n(Grid Avg)',
    '1B': 'Consequential\n(Fossil Avg)',
    '2A': 'Hourly\n(New-Build)',
    '2B': 'Hourly\n(Grid Baseline)',
    '2C': 'Hourly\n(SSS + Tranches)',
    '3A': 'Annual\n(Same-ISO)',
    '3B': 'Annual\n(Cross-Regional)',
};
const STRATEGY_SHORT = {
    '1A': '1A', '1B': '1B', '2A': '2A', '2B': '2B', '2C': '2C', '3A': '3A', '3B': '3B'
};

// Resource colors from chart-colors.js (will use RESOURCE_COLORS global)
function getResourceColor(resource) {
    const map = {
        'solar': '#F59E0B',
        'wind': '#22C55E',
        'offshore_wind': '#009688',
        'hydro': '#0EA5E9',
        'clean_firm': '#6366F1',
        'nuclear_uprate': '#818CF8',
        'ccs': '#64748B',
        'battery': '#C4B5FD',
        'storage': '#EF4444',
        'ldes': '#E91E63',
        'green_h2': '#10B981',
        'geothermal': '#D97706',
        'existing_vre': '#86EFAC',
        'existing_nuclear': '#A5B4FC',
        'new_vre': '#4ADE80',
        'grid_clean': '#94A3B8',
        'sss_allocation': '#CBD5E1',
    };
    // Try RESOURCE_COLORS global first
    if (typeof RESOURCE_COLORS !== 'undefined') {
        const rc = RESOURCE_COLORS;
        if (resource === 'solar') return rc.solar;
        if (resource === 'wind') return rc.wind;
        if (resource === 'hydro') return rc.hydro;
        if (resource === 'clean_firm') return rc.cleanFirm || rc.nuclear;
        if (resource === 'ccs') return rc.ccs;
        if (resource === 'ldes') return rc.ldes;
        if (resource === 'storage') return rc.storage;
        if (resource === 'battery') return rc.battery;
        if (resource === 'geothermal') return rc.geothermal;
        if (resource === 'green_h2') return rc.greenH2;
    }
    return map[resource] || '#9CA3AF';
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
// BALL CLASS — Animated resource ball
// ═══════════════════════════════════════════════════════════════════════════════

class Ball {
    constructor(resource, twh, isExisting, sourceIso) {
        this.resource = resource;
        this.twh = twh;
        this.isExisting = isExisting;
        this.sourceIso = sourceIso;  // null for same-ISO
        this.color = getResourceColor(resource);

        // Position (set by jar packing)
        this.x = 0;
        this.y = 0;
        this.targetX = 0;
        this.targetY = 0;

        // Radius proportional to sqrt(TWh) for area-proportional sizing
        this.radius = 0;  // Set by jar

        // Animation state
        this.opacity = 0;
        this.targetOpacity = 1;
        this.velocity = 0;
        this.entered = false;
    }

    animate(dt) {
        // Spring physics for position
        const springK = 8;
        const damping = 0.7;

        const dx = this.targetX - this.x;
        const dy = this.targetY - this.y;

        this.x += dx * springK * dt;
        this.y += dy * springK * dt;

        // Opacity fade
        const opDiff = this.targetOpacity - this.opacity;
        this.opacity += opDiff * 6 * dt;
        this.opacity = Math.max(0, Math.min(1, this.opacity));

        // Check if settled
        return Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5 || Math.abs(opDiff) > 0.01;
    }

    draw(ctx) {
        if (this.opacity <= 0.01 || this.radius <= 0) return;

        ctx.save();

        if (this.isExisting) {
            // Saturated fill for existing
            ctx.globalAlpha = this.opacity;
            ctx.fillStyle = this.color;
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fill();
        } else {
            // Semi-transparent fill + saturated outline for new
            ctx.globalAlpha = this.opacity * 0.35;
            ctx.fillStyle = this.color;
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fill();

            ctx.globalAlpha = this.opacity;
            ctx.strokeStyle = this.color;
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        ctx.restore();
    }

    hitTest(mx, my) {
        const dx = mx - this.x;
        const dy = my - this.y;
        return (dx * dx + dy * dy) <= (this.radius + 2) * (this.radius + 2);
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// JAR CLASS — Single jar container
// ═══════════════════════════════════════════════════════════════════════════════

class Jar {
    constructor(strategy, iso, x, y, width, height) {
        this.strategy = strategy;
        this.iso = iso;
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.balls = [];
        this.glowing = false;
        this.glowIntensity = 0;
        this.glowPhase = Math.random() * Math.PI * 2;  // offset for pulse
        this.data = null;  // Current data record
    }

    setBalls(dataRecord) {
        this.data = dataRecord;
        const oldBalls = this.balls;
        this.balls = [];

        if (!dataRecord) return;

        // Collect all resources: existing (e), new (n), cross-ISO (x)
        const items = [];

        if (dataRecord.e) {
            for (const [res, twh] of Object.entries(dataRecord.e)) {
                if (twh > 0) items.push({ resource: res, twh, isExisting: true, sourceIso: null });
            }
        }
        if (dataRecord.n) {
            for (const [res, twh] of Object.entries(dataRecord.n)) {
                if (twh > 0) items.push({ resource: res, twh, isExisting: false, sourceIso: null });
            }
        }
        if (dataRecord.x) {
            for (const [srcIso, resources] of Object.entries(dataRecord.x)) {
                for (const [res, twh] of Object.entries(resources)) {
                    if (twh > 0) items.push({ resource: res, twh, isExisting: false, sourceIso: srcIso });
                }
            }
        }

        // Sort: existing first, then by TWh descending
        items.sort((a, b) => {
            if (a.isExisting !== b.isExisting) return a.isExisting ? -1 : 1;
            return b.twh - a.twh;
        });

        // Compute total TWh for radius scaling
        const totalTwh = items.reduce((s, i) => s + i.twh, 0);
        if (totalTwh <= 0) return;

        // Create balls and pack them
        const jarInnerWidth = this.width * 0.75;
        const jarInnerHeight = this.height * 0.65;
        const jarCenterX = this.x + this.width / 2;
        const jarBottom = this.y + this.height * 0.88;

        // Scale factor: largest ball should be ~30% of jar width
        const maxTwh = Math.max(...items.map(i => i.twh));
        const maxRadius = jarInnerWidth * 0.18;
        const minRadius = 3;
        const scaleFactor = maxTwh > 0 ? maxRadius / Math.sqrt(maxTwh) : 1;

        // Create balls
        for (const item of items) {
            const ball = new Ball(item.resource, item.twh, item.isExisting, item.sourceIso);
            ball.radius = Math.max(minRadius, Math.min(maxRadius, Math.sqrt(item.twh) * scaleFactor));
            this.balls.push(ball);
        }

        // Pack balls from bottom up using simple gravity packing
        this.packBalls(jarCenterX, jarBottom, jarInnerWidth, jarInnerHeight);

        // Transfer positions from old balls for smooth animation
        for (const ball of this.balls) {
            // Find matching old ball
            const match = oldBalls.find(ob => ob.resource === ball.resource && ob.isExisting === ball.isExisting);
            if (match) {
                ball.x = match.x;
                ball.y = match.y;
                ball.opacity = match.opacity;
            } else {
                // New ball — start from top, drop in
                ball.x = ball.targetX + (Math.random() - 0.5) * jarInnerWidth * 0.3;
                ball.y = this.y - ball.radius * 2;
                ball.opacity = 0;
            }
            ball.targetOpacity = 1;
            ball.entered = true;
        }

        // Fade out removed balls
        for (const ob of oldBalls) {
            if (!this.balls.find(b => b.resource === ob.resource && b.isExisting === ob.isExisting)) {
                ob.targetOpacity = 0;
                ob.targetY = this.y - ob.radius * 3;  // float up
                this.balls.push(ob);  // keep for animation
            }
        }
    }

    packBalls(cx, bottom, maxWidth, maxHeight) {
        // Simple circle packing: place balls in rows from bottom
        if (this.balls.length === 0) return;

        const padding = 2;
        let currentY = bottom;
        let i = 0;

        while (i < this.balls.length && currentY > (bottom - maxHeight)) {
            // Find how many balls fit in this row
            const rowBalls = [];
            let rowWidth = 0;
            const rowHeight = this.balls[i].radius * 2 + padding;

            while (i < this.balls.length) {
                const ball = this.balls[i];
                const ballWidth = ball.radius * 2 + padding;
                if (rowWidth + ballWidth > maxWidth && rowBalls.length > 0) break;
                rowBalls.push(ball);
                rowWidth += ballWidth;
                i++;
            }

            // Center the row
            currentY -= rowHeight / 2;
            let startX = cx - rowWidth / 2;

            for (const ball of rowBalls) {
                ball.targetX = startX + ball.radius + padding / 2;
                ball.targetY = currentY;
                startX += ball.radius * 2 + padding;
            }

            currentY -= rowHeight / 2 + padding;
        }
    }

    draw(ctx, time) {
        // Draw jar outline (beaker/flask shape)
        const x = this.x;
        const y = this.y;
        const w = this.width;
        const h = this.height;
        const topWidth = w * 0.85;
        const bottomWidth = w * 0.6;
        const topY = y + h * 0.15;
        const bottomY = y + h * 0.9;
        const cx = x + w / 2;
        const cornerR = 4;

        // Glow effect for cross-ISO sources
        if (this.glowing) {
            this.glowIntensity = 0.3 + 0.2 * Math.sin(time * 3 + this.glowPhase);
            ctx.save();
            ctx.shadowColor = getIsoColor(this.iso);
            ctx.shadowBlur = 15 * this.glowIntensity;
            ctx.strokeStyle = getIsoColor(this.iso);
            ctx.globalAlpha = this.glowIntensity;
            ctx.lineWidth = 3;

            ctx.beginPath();
            ctx.moveTo(cx - topWidth / 2, topY);
            ctx.lineTo(cx - bottomWidth / 2, bottomY - cornerR);
            ctx.quadraticCurveTo(cx - bottomWidth / 2, bottomY, cx - bottomWidth / 2 + cornerR, bottomY);
            ctx.lineTo(cx + bottomWidth / 2 - cornerR, bottomY);
            ctx.quadraticCurveTo(cx + bottomWidth / 2, bottomY, cx + bottomWidth / 2, bottomY - cornerR);
            ctx.lineTo(cx + topWidth / 2, topY);
            ctx.stroke();

            ctx.restore();
        }

        // Jar outline
        ctx.save();
        ctx.strokeStyle = '#CBD5E1';
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.8;

        // Draw trapezoidal jar shape
        ctx.beginPath();
        ctx.moveTo(cx - topWidth / 2, topY);
        ctx.lineTo(cx - bottomWidth / 2, bottomY - cornerR);
        ctx.quadraticCurveTo(cx - bottomWidth / 2, bottomY, cx - bottomWidth / 2 + cornerR, bottomY);
        ctx.lineTo(cx + bottomWidth / 2 - cornerR, bottomY);
        ctx.quadraticCurveTo(cx + bottomWidth / 2, bottomY, cx + bottomWidth / 2, bottomY - cornerR);
        ctx.lineTo(cx + topWidth / 2, topY);
        ctx.stroke();

        // Rim at top
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.6;
        ctx.beginPath();
        ctx.moveTo(cx - topWidth / 2 - 3, topY);
        ctx.lineTo(cx + topWidth / 2 + 3, topY);
        ctx.stroke();

        ctx.restore();

        // Draw balls (clipped to jar area)
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(cx - topWidth / 2, topY);
        ctx.lineTo(cx - bottomWidth / 2, bottomY);
        ctx.lineTo(cx + bottomWidth / 2, bottomY);
        ctx.lineTo(cx + topWidth / 2, topY);
        ctx.closePath();
        ctx.clip();

        for (const ball of this.balls) {
            ball.draw(ctx);
        }
        ctx.restore();
    }

    hitTest(mx, my) {
        // Check if mouse is in jar bounds
        if (mx < this.x || mx > this.x + this.width || my < this.y || my > this.y + this.height) {
            return null;
        }
        // Check individual balls
        for (const ball of this.balls) {
            if (ball.hitTest(mx, my)) {
                return { jar: this, ball };
            }
        }
        return { jar: this, ball: null };
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// JAR GRID — Main visualization controller
// ═══════════════════════════════════════════════════════════════════════════════

class JarGrid {
    constructor(canvasId, tooltipId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.tooltipEl = document.getElementById(tooltipId);

        this.participation = 10;  // default
        this.threshold = 90;      // default
        this.data = null;

        this.jars = [];
        this.animating = false;
        this.lastTime = 0;

        // Layout
        this.rowHeaderWidth = 0;
        this.colHeaderHeight = 0;
        this.jarWidth = 0;
        this.jarHeight = 0;

        // Interaction
        this.hoveredJar = null;
        this.hoveredBall = null;

        // Cross-ISO glow tracking
        this.glowingSources = new Map();  // Map<strategyId, Set<sourceIso>>

        this._boundMouseMove = this._onMouseMove.bind(this);
        this._boundMouseLeave = this._onMouseLeave.bind(this);
        this._boundTouchStart = this._onTouchStart.bind(this);
        this._boundResize = this._onResize.bind(this);

        this.canvas.addEventListener('mousemove', this._boundMouseMove);
        this.canvas.addEventListener('mouseleave', this._boundMouseLeave);
        this.canvas.addEventListener('touchstart', this._boundTouchStart, { passive: false });
        window.addEventListener('resize', this._boundResize);

        this._onResize();
    }

    init(deploymentData) {
        this.data = deploymentData;
        this._buildJars();
        this._updateData();
        this._startAnimation();
    }

    setParticipation(pct) {
        if (this.participation === pct) return;
        this.participation = pct;
        this._updateData();
    }

    setThreshold(thr) {
        if (this.threshold === thr) return;
        this.threshold = thr;
        this._updateData();
    }

    _onResize() {
        const dpr = window.devicePixelRatio || 1;
        const container = this.canvas.parentElement;
        const rect = container.getBoundingClientRect();

        // Determine sizes based on viewport
        const isMobile = window.innerWidth < 768;
        const isTablet = window.innerWidth < 1024;

        this.rowHeaderWidth = isMobile ? 60 : (isTablet ? 80 : 120);
        this.colHeaderHeight = isMobile ? 30 : 40;

        const availWidth = rect.width - this.rowHeaderWidth;
        const jarW = Math.floor(availWidth / ISO_LIST.length);
        const jarH = isMobile ? 100 : (isTablet ? 120 : 150);

        this.jarWidth = jarW;
        this.jarHeight = jarH;

        const totalWidth = this.rowHeaderWidth + jarW * ISO_LIST.length;
        const totalHeight = this.colHeaderHeight + jarH * STRATEGIES.length;

        this.canvas.style.width = totalWidth + 'px';
        this.canvas.style.height = totalHeight + 'px';
        this.canvas.width = totalWidth * dpr;
        this.canvas.height = totalHeight * dpr;
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        this._buildJars();
        this._updateData();
    }

    _buildJars() {
        this.jars = [];
        for (let row = 0; row < STRATEGIES.length; row++) {
            for (let col = 0; col < ISO_LIST.length; col++) {
                const x = this.rowHeaderWidth + col * this.jarWidth;
                const y = this.colHeaderHeight + row * this.jarHeight;
                const jar = new Jar(STRATEGIES[row], ISO_LIST[col], x, y, this.jarWidth, this.jarHeight);
                this.jars.push(jar);
            }
        }
    }

    _findClosestKey(available, target) {
        // Find closest available participation/threshold key
        let best = null;
        let bestDist = Infinity;
        for (const key of available) {
            const val = parseFloat(key);
            const dist = Math.abs(val - target);
            if (dist < bestDist) {
                bestDist = dist;
                best = key;
            }
        }
        return best;
    }

    _updateData() {
        if (!this.data) return;

        const partKey = String(this.participation);
        const thrKey = String(this.threshold);

        // Track cross-ISO glow sources
        this.glowingSources.clear();

        // Collect aggregated stats for callback
        let totalProcured = 0;
        let totalCost = 0;
        let totalCO2 = 0;
        let totalBaseline = 0;

        for (const jar of this.jars) {
            const stratData = this.data.data[jar.strategy];
            if (!stratData) { jar.setBalls(null); continue; }

            const isoData = stratData[jar.iso];
            if (!isoData) { jar.setBalls(null); continue; }

            // Find closest participation level
            const availParts = Object.keys(isoData);
            const pk = this._findClosestKey(availParts, this.participation);
            if (!pk) { jar.setBalls(null); continue; }

            const partData = isoData[pk];
            const availThrs = Object.keys(partData);
            const tk = this._findClosestKey(availThrs, this.threshold);
            if (!tk) { jar.setBalls(null); continue; }

            const record = partData[tk];
            jar.setBalls(record);

            // Accumulate stats
            if (record) {
                totalCost += record.tc || 0;
                totalCO2 += record.co2 || 0;
                totalBaseline += record.bl || 0;
            }

            // Track cross-ISO sources for glow
            if (record && record.x) {
                if (!this.glowingSources.has(jar.strategy)) {
                    this.glowingSources.set(jar.strategy, new Set());
                }
                for (const srcIso of Object.keys(record.x)) {
                    this.glowingSources.get(jar.strategy).add(srcIso);
                }
            }
        }

        // Set glow state on jars
        for (const jar of this.jars) {
            const sources = this.glowingSources.get(jar.strategy);
            jar.glowing = sources ? sources.has(jar.iso) : false;
        }

        // Fire stats callback
        if (this.onStatsUpdate) {
            this.onStatsUpdate({
                totalCostB: totalCost / 1000,  // $M to $B
                totalCO2Mt: totalCO2,
                totalBaselineMt: totalBaseline,
            });
        }
    }

    _startAnimation() {
        if (this.animating) return;
        this.animating = true;
        this.lastTime = performance.now();
        this._animationLoop();
    }

    _animationLoop() {
        if (!this.animating) return;

        const now = performance.now();
        const dt = Math.min((now - this.lastTime) / 1000, 0.05);  // cap dt
        this.lastTime = now;

        // Animate all balls
        let needsUpdate = false;
        for (const jar of this.jars) {
            for (let i = jar.balls.length - 1; i >= 0; i--) {
                const ball = jar.balls[i];
                const moving = ball.animate(dt);
                if (moving) needsUpdate = true;

                // Remove fully faded balls
                if (ball.opacity <= 0.01 && ball.targetOpacity <= 0) {
                    jar.balls.splice(i, 1);
                }
            }
        }

        this._draw(now / 1000);

        requestAnimationFrame(() => this._animationLoop());
    }

    _draw(time) {
        const ctx = this.ctx;
        const w = this.canvas.width / (window.devicePixelRatio || 1);
        const h = this.canvas.height / (window.devicePixelRatio || 1);

        ctx.clearRect(0, 0, w, h);

        // Draw column headers (ISOs)
        ctx.save();
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const fontSize = this.jarWidth < 80 ? 10 : 13;
        ctx.font = `600 ${fontSize}px 'Space Grotesk', sans-serif`;

        for (let col = 0; col < ISO_LIST.length; col++) {
            const iso = ISO_LIST[col];
            const x = this.rowHeaderWidth + col * this.jarWidth + this.jarWidth / 2;
            const y = this.colHeaderHeight / 2;

            ctx.fillStyle = getIsoColor(iso);
            ctx.fillText(iso, x, y);

            // Underline
            const tw = ctx.measureText(iso).width;
            ctx.strokeStyle = getIsoColor(iso);
            ctx.lineWidth = 2;
            ctx.globalAlpha = 0.4;
            ctx.beginPath();
            ctx.moveTo(x - tw / 2, y + fontSize / 2 + 2);
            ctx.lineTo(x + tw / 2, y + fontSize / 2 + 2);
            ctx.stroke();
            ctx.globalAlpha = 1;
        }
        ctx.restore();

        // Draw row headers (strategies)
        ctx.save();
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        const rowFontSize = this.rowHeaderWidth < 80 ? 9 : 11;
        ctx.font = `500 ${rowFontSize}px 'DM Sans', sans-serif`;
        ctx.fillStyle = '#334155';

        for (let row = 0; row < STRATEGIES.length; row++) {
            const strat = STRATEGIES[row];
            const y = this.colHeaderHeight + row * this.jarHeight + this.jarHeight / 2;
            const label = STRATEGY_LABELS[strat] || strat;
            const lines = label.split('\n');

            for (let li = 0; li < lines.length; li++) {
                ctx.fillText(
                    lines[li],
                    this.rowHeaderWidth - 8,
                    y + (li - (lines.length - 1) / 2) * (rowFontSize + 2)
                );
            }
        }
        ctx.restore();

        // Draw grid lines
        ctx.save();
        ctx.strokeStyle = '#E2E8F0';
        ctx.lineWidth = 0.5;

        for (let row = 0; row <= STRATEGIES.length; row++) {
            const y = this.colHeaderHeight + row * this.jarHeight;
            ctx.beginPath();
            ctx.moveTo(this.rowHeaderWidth, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }
        for (let col = 0; col <= ISO_LIST.length; col++) {
            const x = this.rowHeaderWidth + col * this.jarWidth;
            ctx.beginPath();
            ctx.moveTo(x, this.colHeaderHeight);
            ctx.lineTo(x, h);
            ctx.stroke();
        }
        ctx.restore();

        // Draw jars and balls
        for (const jar of this.jars) {
            jar.draw(ctx, time);
        }

        // Hover highlight
        if (this.hoveredJar) {
            ctx.save();
            ctx.strokeStyle = '#3B82F6';
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 4]);
            ctx.strokeRect(
                this.hoveredJar.x + 1,
                this.hoveredJar.y + 1,
                this.hoveredJar.width - 2,
                this.hoveredJar.height - 2
            );
            ctx.restore();
        }
    }

    _onMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        let hit = null;
        for (const jar of this.jars) {
            hit = jar.hitTest(mx, my);
            if (hit) break;
        }

        if (hit) {
            this.hoveredJar = hit.jar;
            this.hoveredBall = hit.ball;
            this.canvas.style.cursor = 'pointer';
            this._showTooltip(e.clientX, e.clientY, hit.jar, hit.ball);
        } else {
            this.hoveredJar = null;
            this.hoveredBall = null;
            this.canvas.style.cursor = 'default';
            this._hideTooltip();
        }
    }

    _onMouseLeave() {
        this.hoveredJar = null;
        this.hoveredBall = null;
        this._hideTooltip();
    }

    _onTouchStart(e) {
        const rect = this.canvas.getBoundingClientRect();
        const touch = e.touches[0];
        const mx = touch.clientX - rect.left;
        const my = touch.clientY - rect.top;

        let hit = null;
        for (const jar of this.jars) {
            hit = jar.hitTest(mx, my);
            if (hit) break;
        }

        if (hit) {
            this.hoveredJar = hit.jar;
            this._showTooltip(touch.clientX, touch.clientY, hit.jar, hit.ball);
            e.preventDefault();
        } else {
            this._hideTooltip();
        }
    }

    _showTooltip(px, py, jar, ball) {
        if (!this.tooltipEl || !jar.data) return;

        const data = jar.data;
        let html = `<div class="deployment-tooltip-header">
            <strong>${STRATEGY_LABELS[jar.strategy].replace('\n', ' ')}</strong> —
            <span style="color:${getIsoColor(jar.iso)}">${jar.iso}</span>
        </div>`;

        // Summary stats
        html += `<div class="deployment-tooltip-stats">
            <span>Buyer demand: ${(data.bt || 0).toFixed(1)} TWh</span>
            <span>Cost: $${(data.c || 0).toFixed(0)}/MWh</span>
        </div>`;

        // Resource breakdown
        html += '<div class="deployment-tooltip-breakdown">';

        if (data.e && Object.keys(data.e).length > 0) {
            html += '<div class="tooltip-section-label">Existing Clean</div>';
            for (const [res, twh] of Object.entries(data.e)) {
                const color = getResourceColor(res);
                html += `<div class="tooltip-resource">
                    <span class="tooltip-dot" style="background:${color}"></span>
                    <span>${formatResourceName(res)}</span>
                    <span class="tooltip-twh">${twh.toFixed(1)} TWh</span>
                </div>`;
            }
        }

        if (data.n && Object.keys(data.n).length > 0) {
            html += '<div class="tooltip-section-label">New Build</div>';
            for (const [res, twh] of Object.entries(data.n)) {
                const color = getResourceColor(res);
                html += `<div class="tooltip-resource">
                    <span class="tooltip-dot-hollow" style="border-color:${color}"></span>
                    <span>${formatResourceName(res)}</span>
                    <span class="tooltip-twh">${twh.toFixed(1)} TWh</span>
                </div>`;
            }
        }

        if (data.x && Object.keys(data.x).length > 0) {
            html += '<div class="tooltip-section-label">Cross-ISO Capital Flows</div>';
            for (const [srcIso, resources] of Object.entries(data.x)) {
                for (const [res, twh] of Object.entries(resources)) {
                    html += `<div class="tooltip-resource">
                        <span class="tooltip-iso-badge" style="background:${getIsoColor(srcIso)}20;color:${getIsoColor(srcIso)}">${srcIso}</span>
                        <span>${formatResourceName(res)}</span>
                        <span class="tooltip-twh">${twh.toFixed(1)} TWh</span>
                    </div>`;
                }
            }
        }

        html += '</div>';

        // CO2 reduction
        if (data.co2 > 0) {
            const reductionPct = data.co2r ? data.co2r.toFixed(0) + '%' : '';
            const baselineInfo = data.bl ? ` of ${(data.bl * 1000).toFixed(0)} kt baseline` : '';
            html += `<div class="deployment-tooltip-footer">
                CO₂ reduced: ${(data.co2 * 1000).toFixed(1)} ktCO₂ ${reductionPct ? `(${reductionPct}${baselineInfo})` : ''}
                ${data.mac ? ` · MAC: $${Math.round(data.mac)}/tCO₂` : ''}
            </div>`;
        }

        this.tooltipEl.innerHTML = html;
        this.tooltipEl.style.display = 'block';

        // Position tooltip
        const ttRect = this.tooltipEl.getBoundingClientRect();
        let left = px + 12;
        let top = py - 10;

        if (left + ttRect.width > window.innerWidth - 10) {
            left = px - ttRect.width - 12;
        }
        if (top + ttRect.height > window.innerHeight - 10) {
            top = window.innerHeight - ttRect.height - 10;
        }

        this.tooltipEl.style.left = left + 'px';
        this.tooltipEl.style.top = top + 'px';
    }

    _hideTooltip() {
        if (this.tooltipEl) {
            this.tooltipEl.style.display = 'none';
        }
    }

    getAggregateStats() {
        const stats = {};
        for (const strat of STRATEGIES) {
            let totalExisting = 0, totalNew = 0, totalCross = 0;
            let totalCost = 0, totalCO2 = 0, totalBaseline = 0;
            const resourceBreakdown = { existing: {}, new: {} };

            for (const jar of this.jars) {
                if (jar.strategy !== strat || !jar.data) continue;
                const d = jar.data;

                if (d.e) {
                    for (const [r, t] of Object.entries(d.e)) {
                        totalExisting += t;
                        resourceBreakdown.existing[r] = (resourceBreakdown.existing[r] || 0) + t;
                    }
                }
                if (d.n) {
                    for (const [r, t] of Object.entries(d.n)) {
                        totalNew += t;
                        resourceBreakdown.new[r] = (resourceBreakdown.new[r] || 0) + t;
                    }
                }
                if (d.x) {
                    for (const resources of Object.values(d.x)) {
                        for (const t of Object.values(resources)) {
                            totalCross += t;
                        }
                    }
                }
                totalCost += d.tc || 0;
                totalCO2 += d.co2 || 0;
                totalBaseline += d.bl || 0;
            }

            stats[strat] = {
                existingTwh: totalExisting,
                newTwh: totalNew,
                crossTwh: totalCross,
                totalTwh: totalExisting + totalNew + totalCross,
                totalCostM: totalCost,
                totalCO2Mt: totalCO2,
                totalBaselineMt: totalBaseline,
                co2ReductionPct: totalBaseline > 0 ? (totalCO2 / totalBaseline * 100) : 0,
                mac: totalCO2 > 0 ? (totalCost * 1e6) / (totalCO2 * 1e6) : 0,
                resourceBreakdown,
            };
        }
        return stats;
    }

    dispose() {
        this.animating = false;
        this.canvas.removeEventListener('mousemove', this._boundMouseMove);
        this.canvas.removeEventListener('mouseleave', this._boundMouseLeave);
        this.canvas.removeEventListener('touchstart', this._boundTouchStart);
        window.removeEventListener('resize', this._boundResize);
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

function formatResourceName(key) {
    const names = {
        'solar': 'Solar',
        'wind': 'Wind',
        'offshore_wind': 'Offshore Wind',
        'hydro': 'Hydro',
        'clean_firm': 'Clean Firm',
        'nuclear_uprate': 'Nuclear Uprate',
        'ccs': 'CCS-CCGT',
        'battery': 'Battery',
        'storage': 'Storage',
        'ldes': 'LDES',
        'green_h2': 'Green H₂',
        'geothermal': 'Geothermal',
        'existing_vre': 'Existing VRE/Hydro',
        'existing_nuclear': 'Existing Nuclear',
        'new_vre': 'New VRE',
        'grid_clean': 'Grid Clean Mix',
        'sss_allocation': 'SSS Allocation',
    };
    return names[key] || key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}
