/**
 * Jar Animation Engine v2 — Uniform ball grid visualization
 *
 * Each ball = a fixed share of deployed resources. Balls are uniform size,
 * stacked in orderly rows inside curved-bottom jars. Cross-ISO balls get
 * a glow outline in the source ISO's color.
 *
 * Free credits (SSS allocation, grid baseline credit, existing_vre in 2B)
 * are excluded — only paid procurement is shown.
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

// Resources that are "free credits" — not paid procurement
const FREE_RESOURCES = new Set([
    'sss_allocation', 'grid_clean', 'existing_vre', 'existing_nuclear'
]);

// Approximate annual grid demand per ISO (TWh, 2023-2024)
const GRID_DEMANDS = {
    'CAISO': 280, 'ERCOT': 440, 'PJM': 800,
    'NYISO': 165, 'NEISO': 125, 'MISO': 620, 'SPP': 260,
};

function getResourceColor(resource) {
    const map = {
        'solar': '#F59E0B', 'wind': '#22C55E', 'offshore_wind': '#009688',
        'hydro': '#0EA5E9', 'clean_firm': '#6366F1', 'nuclear_uprate': '#818CF8',
        'ccs': '#64748B', 'battery': '#C4B5FD', 'storage': '#EF4444',
        'ldes': '#E91E63', 'green_h2': '#10B981', 'geothermal': '#D97706',
        'new_vre': '#4ADE80', 'new_build_uprate': '#818CF8',
    };
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
        if (resource === 'nuclear_uprate') return '#818CF8';
        if (resource === 'new_vre') return '#4ADE80';
        if (resource === 'new_build_uprate') return '#818CF8';
    }
    return map[resource] || '#9CA3AF';
}

function getResourceLabel(key) {
    const names = {
        'solar': 'Solar', 'wind': 'Wind', 'offshore_wind': 'Offshore Wind',
        'hydro': 'Hydro', 'clean_firm': 'Clean Firm', 'nuclear_uprate': 'Nuclear Uprate',
        'ccs': 'CCS-CCGT', 'battery': 'Battery', 'storage': 'Storage',
        'ldes': 'LDES', 'green_h2': 'Green H₂', 'geothermal': 'Geothermal',
        'new_vre': 'New VRE', 'new_build_uprate': 'New Uprate',
    };
    return names[key] || key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
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
// BALL CLASS — Uniform-size resource ball
// ═══════════════════════════════════════════════════════════════════════════════

class Ball {
    constructor(resource, isExisting, sourceIso) {
        this.resource = resource;
        this.isExisting = isExisting;
        this.sourceIso = sourceIso;  // null for same-ISO
        this.color = getResourceColor(resource);
        this.glowColor = sourceIso ? getIsoColor(sourceIso) : null;

        // Position (set by jar packing)
        this.x = 0;
        this.y = 0;
        this.targetX = 0;
        this.targetY = 0;
        this.radius = 0;  // Uniform, set by jar

        // Animation
        this.opacity = 0;
        this.targetOpacity = 1;
        this.entered = false;
    }

    animate(dt) {
        const springK = 10;
        const dx = this.targetX - this.x;
        const dy = this.targetY - this.y;
        this.x += dx * springK * dt;
        this.y += dy * springK * dt;

        const opDiff = this.targetOpacity - this.opacity;
        this.opacity += opDiff * 6 * dt;
        this.opacity = Math.max(0, Math.min(1, this.opacity));

        return Math.abs(dx) > 0.3 || Math.abs(dy) > 0.3 || Math.abs(opDiff) > 0.01;
    }

    draw(ctx, time) {
        if (this.opacity <= 0.01 || this.radius <= 0) return;
        ctx.save();

        const r = this.radius;

        // Cross-ISO glow outline
        if (this.glowColor) {
            const pulse = 0.5 + 0.3 * Math.sin(time * 3 + this.x * 0.1);
            ctx.globalAlpha = this.opacity * pulse;
            ctx.shadowColor = this.glowColor;
            ctx.shadowBlur = r * 1.2;
            ctx.strokeStyle = this.glowColor;
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            ctx.arc(this.x, this.y, r + 1, 0, Math.PI * 2);
            ctx.stroke();
            ctx.shadowBlur = 0;
        }

        // Main ball
        if (this.isExisting) {
            // Solid fill for existing/paid resources
            ctx.globalAlpha = this.opacity;
            ctx.fillStyle = this.color;
            ctx.beginPath();
            ctx.arc(this.x, this.y, r, 0, Math.PI * 2);
            ctx.fill();
        } else {
            // Semi-transparent fill + border for new-build
            ctx.globalAlpha = this.opacity * 0.4;
            ctx.fillStyle = this.color;
            ctx.beginPath();
            ctx.arc(this.x, this.y, r, 0, Math.PI * 2);
            ctx.fill();

            ctx.globalAlpha = this.opacity;
            ctx.strokeStyle = this.color;
            ctx.lineWidth = 1.8;
            ctx.stroke();
        }

        ctx.restore();
    }

    hitTest(mx, my) {
        const dx = mx - this.x;
        const dy = my - this.y;
        return (dx * dx + dy * dy) <= (this.radius + 3) * (this.radius + 3);
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// JAR CLASS — Curved-bottom container with orderly ball rows
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
        this.data = null;

        // Jar geometry (computed once, updated on resize)
        this._computeGeometry();
    }

    _computeGeometry() {
        const w = this.width;
        const h = this.height;
        const cx = this.x + w / 2;

        this.cx = cx;
        this.rimY = this.y + h * 0.10;
        this.rimW = w * 0.80;
        this.bodyW = w * 0.72;
        this.curveStartY = this.y + h * 0.68;
        this.bottomY = this.y + h * 0.92;
        this.curveDepth = this.bottomY - this.curveStartY;
    }

    /**
     * Get inner half-width at a given y position (for ball packing)
     */
    getHalfWidthAt(y) {
        if (y <= this.rimY) return this.rimW / 2;
        if (y <= this.curveStartY) {
            // Straight section: linear taper from rimW to bodyW
            const frac = (y - this.rimY) / (this.curveStartY - this.rimY);
            return (this.rimW + (this.bodyW - this.rimW) * frac) / 2;
        }
        // Curved bottom section: cosine falloff
        const frac = (y - this.curveStartY) / (this.bottomY - this.curveStartY);
        return (this.bodyW / 2) * Math.cos(Math.min(frac, 1) * Math.PI / 2);
    }

    setBalls(dataRecord, ballTwh) {
        this.data = dataRecord;
        const oldBalls = this.balls;
        this.balls = [];

        if (!dataRecord || ballTwh <= 0) return;

        // Collect paid resources only (skip free credits)
        const items = [];  // {resource, count, isExisting, sourceIso}

        // Existing paid resources (e bucket)
        if (dataRecord.e) {
            for (const [res, twh] of Object.entries(dataRecord.e)) {
                if (FREE_RESOURCES.has(res) || twh <= 0) continue;
                const count = Math.max(1, Math.round(twh / ballTwh));
                items.push({ resource: res, count, isExisting: true, sourceIso: null, twh });
            }
        }

        // New-build resources (n bucket)
        if (dataRecord.n) {
            for (const [res, twh] of Object.entries(dataRecord.n)) {
                if (FREE_RESOURCES.has(res) || twh <= 0) continue;
                const count = Math.max(1, Math.round(twh / ballTwh));
                items.push({ resource: res, count, isExisting: false, sourceIso: null, twh });
            }
        }

        // Cross-ISO resources (x bucket)
        if (dataRecord.x) {
            for (const [srcIso, resources] of Object.entries(dataRecord.x)) {
                for (const [res, twh] of Object.entries(resources)) {
                    if (twh <= 0) continue;
                    const count = Math.max(1, Math.round(twh / ballTwh));
                    items.push({ resource: res, count, isExisting: false, sourceIso: srcIso, twh });
                }
            }
        }

        if (items.length === 0) return;

        // Sort: existing first, then by count descending for nice visual stacking
        items.sort((a, b) => {
            if (a.isExisting !== b.isExisting) return a.isExisting ? -1 : 1;
            return b.count - a.count;
        });

        // Create ball instances
        for (const item of items) {
            for (let i = 0; i < item.count; i++) {
                this.balls.push(new Ball(item.resource, item.isExisting, item.sourceIso));
            }
        }

        // Cap at max balls to avoid visual overload
        const MAX_BALLS = 70;
        if (this.balls.length > MAX_BALLS) {
            this.balls.length = MAX_BALLS;
        }

        // Compute ball radius: fit ~6-8 balls per row in widest section
        const innerW = this.bodyW * 0.88;
        const ballsPerRow = Math.max(4, Math.min(8, Math.ceil(Math.sqrt(this.balls.length * 1.5))));
        const ballDiameter = innerW / ballsPerRow;
        const ballRadius = Math.max(2.5, Math.min(ballDiameter / 2 - 1, 9));

        for (const ball of this.balls) {
            ball.radius = ballRadius;
        }

        // Pack balls in orderly rows from bottom up
        this._packRows(ballRadius);

        // Animate: transfer from old balls or drop in
        for (const ball of this.balls) {
            const match = oldBalls.find(ob =>
                ob.resource === ball.resource &&
                ob.isExisting === ball.isExisting &&
                ob.sourceIso === ball.sourceIso &&
                ob.opacity > 0.5
            );
            if (match) {
                ball.x = match.x;
                ball.y = match.y;
                ball.opacity = match.opacity;
                // Remove from oldBalls so each only matches once
                oldBalls.splice(oldBalls.indexOf(match), 1);
            } else {
                ball.x = ball.targetX + (Math.random() - 0.5) * 10;
                ball.y = this.rimY - ballRadius * 3;
                ball.opacity = 0;
            }
            ball.targetOpacity = 1;
            ball.entered = true;
        }

        // Fade out removed balls
        for (const ob of oldBalls) {
            if (ob.opacity > 0.05) {
                ob.targetOpacity = 0;
                ob.targetY = this.rimY - ob.radius * 4;
                this.balls.push(ob);
            }
        }
    }

    _packRows(ballR) {
        if (this.balls.length === 0) return;

        const diameter = ballR * 2;
        const padding = 1.5;
        const step = diameter + padding;

        // Start packing from bottom of jar upward
        let currentY = this.bottomY - ballR - 2;
        let idx = 0;

        while (idx < this.balls.length && currentY > this.rimY + ballR) {
            // How wide is the jar at this y?
            const halfW = this.getHalfWidthAt(currentY) - ballR - 2;
            if (halfW < ballR) {
                currentY -= step * 0.5;
                continue;
            }

            // How many balls fit in this row?
            const rowCapacity = Math.max(1, Math.floor((halfW * 2) / step));
            const ballsThisRow = Math.min(rowCapacity, this.balls.length - idx);

            // Center the row
            const rowWidth = ballsThisRow * step - padding;
            let startX = this.cx - rowWidth / 2 + ballR;

            for (let i = 0; i < ballsThisRow; i++) {
                this.balls[idx].targetX = startX + i * step;
                this.balls[idx].targetY = currentY;
                idx++;
            }

            currentY -= step;
        }

        // Any remaining balls that don't fit: stack at top
        while (idx < this.balls.length) {
            this.balls[idx].targetX = this.cx;
            this.balls[idx].targetY = this.rimY + ballR + 2;
            idx++;
        }
    }

    draw(ctx, time) {
        const cx = this.cx;
        const rimY = this.rimY;
        const rimW = this.rimW;
        const bodyW = this.bodyW;
        const curveStartY = this.curveStartY;
        const bottomY = this.bottomY;
        const lipExtra = 4;

        // ---- Subtle fill gradient inside jar ----
        ctx.save();
        const fillGrad = ctx.createLinearGradient(cx, rimY, cx, bottomY);
        fillGrad.addColorStop(0, 'rgba(241, 245, 249, 0.3)');
        fillGrad.addColorStop(1, 'rgba(226, 232, 240, 0.15)');
        ctx.fillStyle = fillGrad;

        ctx.beginPath();
        ctx.moveTo(cx - rimW / 2, rimY);
        ctx.lineTo(cx - bodyW / 2, curveStartY);
        ctx.quadraticCurveTo(cx - bodyW / 2, bottomY, cx, bottomY);
        ctx.quadraticCurveTo(cx + bodyW / 2, bottomY, cx + bodyW / 2, curveStartY);
        ctx.lineTo(cx + rimW / 2, rimY);
        ctx.closePath();
        ctx.fill();
        ctx.restore();

        // ---- Clip region for balls ----
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(cx - rimW / 2 + 2, rimY);
        ctx.lineTo(cx - bodyW / 2 + 2, curveStartY);
        ctx.quadraticCurveTo(cx - bodyW / 2 + 2, bottomY - 2, cx, bottomY - 2);
        ctx.quadraticCurveTo(cx + bodyW / 2 - 2, bottomY - 2, cx + bodyW / 2 - 2, curveStartY);
        ctx.lineTo(cx + rimW / 2 - 2, rimY);
        ctx.closePath();
        ctx.clip();

        // Draw balls
        for (const ball of this.balls) {
            ball.draw(ctx, time);
        }
        ctx.restore();

        // ---- Jar outline (drawn on top) ----
        ctx.save();
        ctx.strokeStyle = '#94A3B8';
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.7;

        // Main jar body
        ctx.beginPath();
        ctx.moveTo(cx - rimW / 2, rimY);
        ctx.lineTo(cx - bodyW / 2, curveStartY);
        ctx.quadraticCurveTo(cx - bodyW / 2, bottomY, cx, bottomY);
        ctx.quadraticCurveTo(cx + bodyW / 2, bottomY, cx + bodyW / 2, curveStartY);
        ctx.lineTo(cx + rimW / 2, rimY);
        ctx.stroke();

        // Rim lip
        ctx.lineWidth = 2.5;
        ctx.globalAlpha = 0.55;
        ctx.beginPath();
        ctx.moveTo(cx - rimW / 2 - lipExtra, rimY);
        ctx.lineTo(cx + rimW / 2 + lipExtra, rimY);
        ctx.stroke();

        // Glass reflection (subtle arc on left side)
        ctx.globalAlpha = 0.12;
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2;
        const reflectX = cx - bodyW * 0.28;
        ctx.beginPath();
        ctx.moveTo(reflectX, rimY + 8);
        ctx.lineTo(reflectX - 2, curveStartY - 10);
        ctx.stroke();

        ctx.restore();
    }

    hitTest(mx, my) {
        if (mx < this.x || mx > this.x + this.width ||
            my < this.y || my > this.y + this.height) return null;
        for (const ball of this.balls) {
            if (ball.hitTest(mx, my)) return { jar: this, ball };
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

        this.participation = 10;
        this.threshold = 90;
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

        const isMobile = window.innerWidth < 768;
        const isTablet = window.innerWidth < 1024;

        this.rowHeaderWidth = isMobile ? 55 : (isTablet ? 75 : 110);
        this.colHeaderHeight = isMobile ? 28 : 38;

        const availWidth = rect.width - this.rowHeaderWidth;
        const jarW = Math.max(55, Math.floor(availWidth / ISO_LIST.length));
        const jarH = isMobile ? 110 : (isTablet ? 130 : 160);

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
                this.jars.push(new Jar(STRATEGIES[row], ISO_LIST[col], x, y, this.jarWidth, this.jarHeight));
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

    /**
     * Sum paid TWh from a data record (excluding free credits)
     */
    _sumPaidTwh(record) {
        let total = 0;
        if (record.e) {
            for (const [res, twh] of Object.entries(record.e)) {
                if (!FREE_RESOURCES.has(res) && twh > 0) total += twh;
            }
        }
        if (record.n) {
            for (const [res, twh] of Object.entries(record.n)) {
                if (!FREE_RESOURCES.has(res) && twh > 0) total += twh;
            }
        }
        if (record.x) {
            for (const resources of Object.values(record.x)) {
                for (const twh of Object.values(resources)) {
                    if (twh > 0) total += twh;
                }
            }
        }
        return total;
    }

    _updateData() {
        if (!this.data) return;

        // First pass: find all records and compute max paid TWh for global ball sizing
        const records = [];
        let maxPaidTwh = 0;

        for (const jar of this.jars) {
            const stratData = this.data.data[jar.strategy];
            if (!stratData) { records.push(null); continue; }
            const isoData = stratData[jar.iso];
            if (!isoData) { records.push(null); continue; }

            const pk = this._findClosestKey(Object.keys(isoData), this.participation);
            if (!pk) { records.push(null); continue; }
            const tk = this._findClosestKey(Object.keys(isoData[pk]), this.threshold);
            if (!tk) { records.push(null); continue; }

            const record = isoData[pk][tk];
            records.push(record);

            if (record) {
                const paidTwh = this._sumPaidTwh(record);
                if (paidTwh > maxPaidTwh) maxPaidTwh = paidTwh;
            }
        }

        // Compute ball TWh: target ~40 balls for the most-filled jar
        const TARGET_MAX_BALLS = 45;
        const ballTwh = maxPaidTwh > 0 ? maxPaidTwh / TARGET_MAX_BALLS : 1;

        // Second pass: set balls on each jar
        let totalCO2 = 0;
        let totalBuyerDemand = 0;

        for (let i = 0; i < this.jars.length; i++) {
            const jar = this.jars[i];
            const record = records[i];
            jar.setBalls(record, ballTwh);

            if (record) {
                totalCO2 += record.co2 || 0;
                totalBuyerDemand += record.bt || 0;
            }
        }

        // Fire stats callback (deployment-focused, no cost)
        if (this.onStatsUpdate) {
            // Count total paid TWh across all jars for current strategy
            let totalPaidTwh = 0;
            for (let i = 0; i < records.length; i++) {
                if (records[i]) totalPaidTwh += this._sumPaidTwh(records[i]);
            }
            this.onStatsUpdate({
                totalPaidTwh,
                totalCO2Mt: totalCO2,
                ballTwh,
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
        const dt = Math.min((now - this.lastTime) / 1000, 0.05);
        this.lastTime = now;

        let needsUpdate = false;
        for (const jar of this.jars) {
            for (let i = jar.balls.length - 1; i >= 0; i--) {
                const ball = jar.balls[i];
                if (ball.animate(dt)) needsUpdate = true;
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

        // Column headers (ISOs)
        ctx.save();
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const fontSize = this.jarWidth < 70 ? 9 : (this.jarWidth < 90 ? 11 : 13);
        ctx.font = `600 ${fontSize}px 'Space Grotesk', sans-serif`;

        for (let col = 0; col < ISO_LIST.length; col++) {
            const iso = ISO_LIST[col];
            const x = this.rowHeaderWidth + col * this.jarWidth + this.jarWidth / 2;
            const y = this.colHeaderHeight / 2;
            ctx.fillStyle = getIsoColor(iso);
            ctx.fillText(iso, x, y);
        }
        ctx.restore();

        // Row headers (strategies)
        ctx.save();
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        const rowFS = this.rowHeaderWidth < 65 ? 8 : (this.rowHeaderWidth < 80 ? 9 : 11);
        ctx.font = `500 ${rowFS}px 'DM Sans', sans-serif`;
        ctx.fillStyle = '#334155';

        for (let row = 0; row < STRATEGIES.length; row++) {
            const strat = STRATEGIES[row];
            const y = this.colHeaderHeight + row * this.jarHeight + this.jarHeight / 2;
            const lines = (STRATEGY_LABELS[strat] || strat).split('\n');
            for (let li = 0; li < lines.length; li++) {
                ctx.fillText(lines[li], this.rowHeaderWidth - 6, y + (li - (lines.length - 1) / 2) * (rowFS + 2));
            }
        }
        ctx.restore();

        // Faint grid lines
        ctx.save();
        ctx.strokeStyle = '#E2E8F0';
        ctx.lineWidth = 0.5;
        for (let row = 0; row <= STRATEGIES.length; row++) {
            const y = this.colHeaderHeight + row * this.jarHeight;
            ctx.beginPath(); ctx.moveTo(this.rowHeaderWidth, y); ctx.lineTo(w, y); ctx.stroke();
        }
        for (let col = 0; col <= ISO_LIST.length; col++) {
            const x = this.rowHeaderWidth + col * this.jarWidth;
            ctx.beginPath(); ctx.moveTo(x, this.colHeaderHeight); ctx.lineTo(x, h); ctx.stroke();
        }
        ctx.restore();

        // Jars and balls
        for (const jar of this.jars) {
            jar.draw(ctx, time);
        }

        // Hover highlight
        if (this.hoveredJar) {
            ctx.save();
            ctx.strokeStyle = '#3B82F6';
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 4]);
            ctx.strokeRect(this.hoveredJar.x + 1, this.hoveredJar.y + 1, this.hoveredJar.width - 2, this.hoveredJar.height - 2);
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

        // Buyer demand
        html += `<div class="deployment-tooltip-stats">
            <span>Buyer demand: ${(data.bt || 0).toFixed(1)} TWh</span>
            ${data.co2r ? `<span>CO₂ reduction: ${data.co2r}%</span>` : ''}
        </div>`;

        // Resource breakdown — paid resources only
        html += '<div class="deployment-tooltip-breakdown">';

        if (data.e) {
            const paidExisting = Object.entries(data.e).filter(([r]) => !FREE_RESOURCES.has(r));
            if (paidExisting.length > 0) {
                html += '<div class="tooltip-section-label">Existing Paid</div>';
                for (const [res, twh] of paidExisting) {
                    if (twh <= 0) continue;
                    html += `<div class="tooltip-resource">
                        <span class="tooltip-dot" style="background:${getResourceColor(res)}"></span>
                        <span>${getResourceLabel(res)}</span>
                        <span class="tooltip-twh">${twh.toFixed(1)} TWh</span>
                    </div>`;
                }
            }
        }

        if (data.n) {
            const paidNew = Object.entries(data.n).filter(([r]) => !FREE_RESOURCES.has(r));
            if (paidNew.length > 0) {
                html += '<div class="tooltip-section-label">New Build</div>';
                for (const [res, twh] of paidNew) {
                    if (twh <= 0) continue;
                    html += `<div class="tooltip-resource">
                        <span class="tooltip-dot-hollow" style="border-color:${getResourceColor(res)}"></span>
                        <span>${getResourceLabel(res)}</span>
                        <span class="tooltip-twh">${twh.toFixed(1)} TWh</span>
                    </div>`;
                }
            }
        }

        if (data.x && Object.keys(data.x).length > 0) {
            html += '<div class="tooltip-section-label">Cross-ISO Procurement</div>';
            for (const [srcIso, resources] of Object.entries(data.x)) {
                for (const [res, twh] of Object.entries(resources)) {
                    if (twh <= 0) continue;
                    html += `<div class="tooltip-resource">
                        <span class="tooltip-iso-badge" style="background:${getIsoColor(srcIso)}20;color:${getIsoColor(srcIso)}">${srcIso}</span>
                        <span>${getResourceLabel(res)}</span>
                        <span class="tooltip-twh">${twh.toFixed(1)} TWh</span>
                    </div>`;
                }
            }
        }

        html += '</div>';

        // CO2 info
        if (data.co2 > 0) {
            html += `<div class="deployment-tooltip-footer">
                CO₂ reduced: ${(data.co2 * 1000).toFixed(1)} ktCO₂
                ${data.co2r ? `(${data.co2r}% of baseline)` : ''}
            </div>`;
        }

        // Hovered ball detail
        if (ball && ball.opacity > 0.5) {
            html += `<div class="deployment-tooltip-footer" style="margin-top:4px;font-style:italic">
                ${getResourceLabel(ball.resource)}${ball.isExisting ? ' (existing)' : ' (new)'}${ball.sourceIso ? ` from ${ball.sourceIso}` : ''}
            </div>`;
        }

        this.tooltipEl.innerHTML = html;
        this.tooltipEl.style.display = 'block';

        const ttRect = this.tooltipEl.getBoundingClientRect();
        let left = px + 12;
        let top = py - 10;
        if (left + ttRect.width > window.innerWidth - 10) left = px - ttRect.width - 12;
        if (top + ttRect.height > window.innerHeight - 10) top = window.innerHeight - ttRect.height - 10;
        this.tooltipEl.style.left = left + 'px';
        this.tooltipEl.style.top = top + 'px';
    }

    _hideTooltip() {
        if (this.tooltipEl) this.tooltipEl.style.display = 'none';
    }

    getAggregateStats() {
        const stats = {};
        for (const strat of STRATEGIES) {
            let totalPaid = 0, totalNew = 0, totalCross = 0, totalExisting = 0;
            let totalCO2 = 0, totalBaseline = 0;
            const resourceBreakdown = { existing: {}, new: {} };

            for (const jar of this.jars) {
                if (jar.strategy !== strat || !jar.data) continue;
                const d = jar.data;

                if (d.e) {
                    for (const [r, t] of Object.entries(d.e)) {
                        if (FREE_RESOURCES.has(r) || t <= 0) continue;
                        totalExisting += t;
                        resourceBreakdown.existing[r] = (resourceBreakdown.existing[r] || 0) + t;
                    }
                }
                if (d.n) {
                    for (const [r, t] of Object.entries(d.n)) {
                        if (FREE_RESOURCES.has(r) || t <= 0) continue;
                        totalNew += t;
                        resourceBreakdown.new[r] = (resourceBreakdown.new[r] || 0) + t;
                    }
                }
                if (d.x) {
                    for (const resources of Object.values(d.x)) {
                        for (const t of Object.values(resources)) {
                            if (t > 0) totalCross += t;
                        }
                    }
                }
                totalCO2 += d.co2 || 0;
                totalBaseline += d.bl || 0;
            }

            totalPaid = totalExisting + totalNew + totalCross;

            stats[strat] = {
                existingTwh: totalExisting,
                newTwh: totalNew,
                crossTwh: totalCross,
                totalTwh: totalPaid,
                totalCO2Mt: totalCO2,
                totalBaselineMt: totalBaseline,
                co2ReductionPct: totalBaseline > 0 ? (totalCO2 / totalBaseline * 100) : 0,
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
