/**
 * Jar Animation Engine v3 — Grid-level impact visualization
 *
 * Each jar shows what happens to an ISO's GRID under a procurement strategy.
 * At 0% participation: baseline grid mix (solid balls).
 * As participation increases: new procurement adds to baseline.
 *
 * Visual tiers:
 * - Solid fill: grid baseline clean (always present)
 * - Transparent fill + saturated outline: existing clean claimed by buyers
 * - Transparent fill only: new-build procurement
 * - Above rim with stripe: curtailed energy
 *
 * Glow semantics: in SOURCE ISO jar, balls serving external buyers glow
 * with the BUYER ISO's color.
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


function getResourceColor(resource) {
    const map = {
        'solar': '#F59E0B', 'wind': '#22C55E', 'offshore_wind': '#009688',
        'hydro': '#0EA5E9', 'clean_firm': '#6366F1', 'nuclear_uprate': '#818CF8',
        'ccs': '#64748B', 'battery': '#C4B5FD', 'storage': '#EF4444',
        'ldes': '#E91E63', 'green_h2': '#10B981', 'geothermal': '#D97706',
        'new_vre': '#4ADE80', 'new_build_uprate': '#818CF8',
        'existing_vre': '#86EFAC', 'existing_nuclear': '#A5B4FC',
        'existing_recs': '#86EFAC', 'cross_regional_recs': '#4ADE80',
        'new_build_vre': '#4ADE80',
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
        'existing_vre': 'Existing VRE/Hydro', 'existing_nuclear': 'Existing Nuclear',
        'existing_recs': 'Existing RECs', 'cross_regional_recs': 'Cross-Regional RECs',
        'new_build_vre': 'New Build VRE',
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
// BALL CLASS — Uniform-size resource ball with grid-level visual tiers
// ═══════════════════════════════════════════════════════════════════════════════

class Ball {
    /**
     * @param {string} resource - resource type
     * @param {string} tier - 'baseline'|'claimed'|'new'|'curtailed'
     * @param {string|null} glowIso - ISO color for glow (buyer ISO for source jars)
     */
    constructor(resource, tier, glowIso = null) {
        this.resource = resource;
        this.tier = tier;  // 'baseline', 'claimed', 'new', 'curtailed'
        this.glowIso = glowIso;
        this.color = getResourceColor(resource);
        this.glowColor = glowIso ? getIsoColor(glowIso) : null;

        // Position (set by jar packing)
        this.x = 0;
        this.y = 0;
        this.targetX = 0;
        this.targetY = 0;
        this.radius = 0;

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

        if (this.tier === 'baseline') {
            // Solid fill — grid baseline clean
            ctx.globalAlpha = this.opacity;
            ctx.fillStyle = this.color;
            ctx.beginPath();
            ctx.arc(this.x, this.y, r, 0, Math.PI * 2);
            ctx.fill();
        } else if (this.tier === 'claimed') {
            // Transparent fill + saturated outline — existing clean claimed by buyers
            ctx.globalAlpha = this.opacity * 0.25;
            ctx.fillStyle = this.color;
            ctx.beginPath();
            ctx.arc(this.x, this.y, r, 0, Math.PI * 2);
            ctx.fill();

            ctx.globalAlpha = this.opacity;
            ctx.strokeStyle = this.color;
            ctx.lineWidth = 2.0;
            ctx.stroke();
        } else if (this.tier === 'curtailed') {
            // Curtailed — diagonal stripe pattern above rim
            ctx.globalAlpha = this.opacity * 0.20;
            ctx.fillStyle = this.color;
            ctx.beginPath();
            ctx.arc(this.x, this.y, r, 0, Math.PI * 2);
            ctx.fill();

            // Diagonal stripes
            ctx.globalAlpha = this.opacity * 0.5;
            ctx.strokeStyle = this.color;
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 2]);
            ctx.beginPath();
            ctx.moveTo(this.x - r * 0.7, this.y + r * 0.7);
            ctx.lineTo(this.x + r * 0.7, this.y - r * 0.7);
            ctx.moveTo(this.x - r * 0.3, this.y + r * 0.7);
            ctx.lineTo(this.x + r * 0.7, this.y - r * 0.3);
            ctx.moveTo(this.x - r * 0.7, this.y + r * 0.3);
            ctx.lineTo(this.x + r * 0.3, this.y - r * 0.7);
            ctx.stroke();
            ctx.setLineDash([]);

            // Dashed outline
            ctx.globalAlpha = this.opacity * 0.6;
            ctx.strokeStyle = this.color;
            ctx.lineWidth = 1.5;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.arc(this.x, this.y, r, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);
        } else {
            // Tier 'new' — transparent fill only (new build)
            ctx.globalAlpha = this.opacity * 0.35;
            ctx.fillStyle = this.color;
            ctx.beginPath();
            ctx.arc(this.x, this.y, r, 0, Math.PI * 2);
            ctx.fill();
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
// JAR CLASS — Grid-level view with baseline + procurement + curtailment
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
        this.gridBaseline = null;
        this._computeGeometry();
    }

    _computeGeometry() {
        const w = this.width;
        const h = this.height;
        this.cx = this.x + w / 2;
        this.rimY = this.y + h * 0.08;
        this.rimW = w * 0.82;
        this.bodyW = w * 0.76;
        this.bottomY = this.y + h * 0.92;
        this.innerH = this.bottomY - this.rimY;
    }

    getHalfWidthAt(y) {
        if (y <= this.rimY) return this.rimW / 2;
        const rimZone = this.rimY + this.innerH * 0.04;
        if (y <= rimZone) {
            const frac = (y - this.rimY) / (rimZone - this.rimY);
            return (this.rimW + (this.bodyW - this.rimW) * frac) / 2;
        }
        return this.bodyW / 2;
    }

    setBalls(dataRecord, ballTwh, gridBaseline, crossIsoFlows) {
        this.data = dataRecord;
        this.gridBaseline = gridBaseline;
        const oldBalls = this.balls;
        this.balls = [];

        const iso = this.iso;
        const bl = gridBaseline || {};
        const gridDemand = GRID_DEMANDS[iso] || 300;

        // === Build ball list ===
        const items = [];

        // 1. Grid baseline balls (always present — solid fill)
        for (const [res, pct] of Object.entries(bl)) {
            if (res === 'totalPct' || pct <= 0) continue;
            const twh = pct / 100 * gridDemand;
            const count = Math.max(1, Math.round(twh / ballTwh));
            items.push({ resource: res, count, tier: 'baseline', glowIso: null, twh });
        }

        // 2. Procurement balls (from data record)
        if (dataRecord) {
            // Existing claimed
            if (dataRecord.e) {
                for (const [res, twh] of Object.entries(dataRecord.e)) {
                    if (twh <= 0) continue;
                    const count = Math.max(1, Math.round(twh / ballTwh));
                    items.push({ resource: res, count, tier: 'claimed', glowIso: null, twh });
                }
            }
            // New build
            if (dataRecord.n) {
                for (const [res, twh] of Object.entries(dataRecord.n)) {
                    if (twh <= 0) continue;
                    const count = Math.max(1, Math.round(twh / ballTwh));
                    items.push({ resource: res, count, tier: 'new', glowIso: null, twh });
                }
            }
            // Cross-ISO: balls appear in SOURCE jar, glowing with BUYER color
            // dataRecord.x has {srcIso: {resource: twh}} — these are purchases FROM srcIso
            // In the buyer's jar, we DON'T show cross-ISO balls (they're in the source jar)
            // Instead, we need crossIsoFlows to know what THIS iso's resources serve externally
        }

        // 3. Cross-ISO glow: resources in THIS jar that serve external buyers
        if (crossIsoFlows) {
            for (const [buyerIso, resources] of Object.entries(crossIsoFlows)) {
                for (const [res, twh] of Object.entries(resources)) {
                    if (twh <= 0) continue;
                    const count = Math.max(1, Math.round(twh / ballTwh));
                    items.push({ resource: res, count, tier: 'new', glowIso: buyerIso, twh });
                }
            }
        }

        // 4. Curtailment balls (above rim)
        if (dataRecord && dataRecord.curtTwh > 0) {
            const curtCount = Math.max(1, Math.round(dataRecord.curtTwh / ballTwh));
            // Distribute curtailment across VRE resources proportionally
            const vreResources = ['solar', 'wind'];
            const curtPerRes = Math.ceil(curtCount / vreResources.length);
            for (const res of vreResources) {
                items.push({
                    resource: res,
                    count: curtPerRes,
                    tier: 'curtailed',
                    glowIso: null,
                    twh: dataRecord.curtTwh / vreResources.length,
                });
            }
        }

        if (items.length === 0) return;

        // Sort: baseline at bottom, then claimed, then new, curtailed last (on top)
        const tierOrder = { 'baseline': 0, 'claimed': 1, 'new': 2, 'curtailed': 3 };
        items.sort((a, b) => {
            const ta = tierOrder[a.tier] || 0;
            const tb = tierOrder[b.tier] || 0;
            if (ta !== tb) return ta - tb;
            return b.count - a.count;
        });

        for (const item of items) {
            for (let i = 0; i < item.count; i++) {
                this.balls.push(new Ball(item.resource, item.tier, item.glowIso));
            }
        }

        // Cap at 120 balls (100 = 100% grid + up to 20 curtailment)
        const MAX_BALLS = 120;
        if (this.balls.length > MAX_BALLS) this.balls.length = MAX_BALLS;

        // Ball radius
        const innerW = this.bodyW * 0.90;
        const ballsPerRow = Math.max(4, Math.min(8, Math.ceil(Math.sqrt(this.balls.length * 1.5))));
        const ballDiameter = innerW / ballsPerRow;
        const ballRadius = Math.max(2.5, Math.min(ballDiameter / 2 - 1, 9));

        for (const ball of this.balls) ball.radius = ballRadius;

        this._packRows(ballRadius);

        // Animate transitions
        for (const ball of this.balls) {
            const match = oldBalls.find(ob =>
                ob.resource === ball.resource &&
                ob.tier === ball.tier &&
                ob.glowIso === ball.glowIso &&
                ob.opacity > 0.5
            );
            if (match) {
                ball.x = match.x;
                ball.y = match.y;
                ball.opacity = match.opacity;
                oldBalls.splice(oldBalls.indexOf(match), 1);
            } else {
                ball.x = ball.targetX + (Math.random() - 0.5) * 10;
                ball.y = this.rimY - ballRadius * 3;
                ball.opacity = 0;
            }
            ball.targetOpacity = 1;
            ball.entered = true;
        }

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

        // Separate curtailed balls from normal balls
        const normalBalls = this.balls.filter(b => b.tier !== 'curtailed');
        const curtailedBalls = this.balls.filter(b => b.tier === 'curtailed');

        // Pack normal balls bottom-up inside jar
        let currentY = this.bottomY - ballR - 2;
        let idx = 0;

        while (idx < normalBalls.length && currentY > this.rimY + ballR) {
            const halfW = this.getHalfWidthAt(currentY) - ballR - 2;
            if (halfW < ballR) { currentY -= step * 0.5; continue; }

            const rowCapacity = Math.max(1, Math.floor((halfW * 2) / step));
            const ballsThisRow = Math.min(rowCapacity, normalBalls.length - idx);
            const rowWidth = ballsThisRow * step - padding;
            let startX = this.cx - rowWidth / 2 + ballR;

            for (let i = 0; i < ballsThisRow; i++) {
                normalBalls[idx].targetX = startX + i * step;
                normalBalls[idx].targetY = currentY;
                idx++;
            }
            currentY -= step;
        }

        // Overflow normal balls at rim
        while (idx < normalBalls.length) {
            normalBalls[idx].targetX = this.cx;
            normalBalls[idx].targetY = this.rimY + ballR + 2;
            idx++;
        }

        // Pack curtailed balls ABOVE rim (overflow area)
        if (curtailedBalls.length > 0) {
            let curtY = this.rimY - ballR - 2;
            let ci = 0;
            while (ci < curtailedBalls.length) {
                const halfW = this.rimW / 2 - ballR - 2;
                const rowCapacity = Math.max(1, Math.floor((halfW * 2) / step));
                const ballsThisRow = Math.min(rowCapacity, curtailedBalls.length - ci);
                const rowWidth = ballsThisRow * step - padding;
                let startX = this.cx - rowWidth / 2 + ballR;

                for (let i = 0; i < ballsThisRow; i++) {
                    curtailedBalls[ci].targetX = startX + i * step;
                    curtailedBalls[ci].targetY = curtY;
                    ci++;
                }
                curtY -= step;
            }
        }
    }

    draw(ctx, time) {
        const cx = this.cx;
        const rimY = this.rimY;
        const rimW = this.rimW;
        const bodyW = this.bodyW;
        const bottomY = this.bottomY;
        const lipExtra = 4;

        // ---- Subtle fill gradient ----
        ctx.save();
        const fillGrad = ctx.createLinearGradient(cx, rimY, cx, bottomY);
        fillGrad.addColorStop(0, 'rgba(241, 245, 249, 0.3)');
        fillGrad.addColorStop(1, 'rgba(226, 232, 240, 0.15)');
        ctx.fillStyle = fillGrad;

        ctx.beginPath();
        ctx.moveTo(cx - rimW / 2, rimY);
        ctx.lineTo(cx - bodyW / 2, bottomY);
        ctx.lineTo(cx + bodyW / 2, bottomY);
        ctx.lineTo(cx + rimW / 2, rimY);
        ctx.closePath();
        ctx.fill();
        ctx.restore();

        // ---- Draw balls (NO clipping for curtailed above rim) ----
        ctx.save();

        // Draw normal balls with jar clip
        const normalBalls = this.balls.filter(b => b.tier !== 'curtailed');
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(cx - rimW / 2 + 2, rimY);
        ctx.lineTo(cx - bodyW / 2 + 2, bottomY);
        ctx.lineTo(cx + bodyW / 2 - 2, bottomY);
        ctx.lineTo(cx + rimW / 2 - 2, rimY);
        ctx.closePath();
        ctx.clip();
        for (const ball of normalBalls) ball.draw(ctx, time);
        ctx.restore();

        // Draw curtailed balls WITHOUT clipping (they overflow above rim)
        const curtailedBalls = this.balls.filter(b => b.tier === 'curtailed');
        for (const ball of curtailedBalls) ball.draw(ctx, time);

        ctx.restore();

        // ---- Jar outline ----
        ctx.save();
        ctx.strokeStyle = '#94A3B8';
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.7;

        ctx.beginPath();
        ctx.moveTo(cx - rimW / 2, rimY);
        ctx.lineTo(cx - bodyW / 2, bottomY);
        ctx.lineTo(cx + bodyW / 2, bottomY);
        ctx.lineTo(cx + rimW / 2, rimY);
        ctx.stroke();

        // Rim lip (100% line)
        ctx.lineWidth = 2.5;
        ctx.globalAlpha = 0.55;
        ctx.beginPath();
        ctx.moveTo(cx - rimW / 2 - lipExtra, rimY);
        ctx.lineTo(cx + rimW / 2 + lipExtra, rimY);
        ctx.stroke();

        // Glass reflection
        ctx.globalAlpha = 0.12;
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2;
        const reflectX = cx - bodyW * 0.28;
        ctx.beginPath();
        ctx.moveTo(reflectX + 2, rimY + 8);
        ctx.lineTo(reflectX, bottomY - 10);
        ctx.stroke();

        // Hourly match score label below jar (or grid clean %)
        const hms = this.data && this.data.hms;
        const gridCleanPct = this.data && this.data.gridCleanPct;
        const displayPct = hms != null ? hms : gridCleanPct;

        if (displayPct != null && displayPct >= 0.5) {
            ctx.globalAlpha = 0.65;
            ctx.fillStyle = '#1E293B';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            const fs = Math.max(7, Math.min(10, this.width * 0.08));
            ctx.font = `600 ${fs}px 'DM Sans', sans-serif`;
            const label = hms != null ? `${Math.round(displayPct)}% HMS` : `${Math.round(displayPct)}%`;
            ctx.fillText(label, cx, bottomY + 2);
        } else if (this.gridBaseline) {
            // Show baseline clean % when no data
            const blPct = this.gridBaseline.totalPct || 0;
            if (blPct > 0) {
                ctx.globalAlpha = 0.45;
                ctx.fillStyle = '#64748B';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                const fs = Math.max(7, Math.min(10, this.width * 0.08));
                ctx.font = `600 ${fs}px 'DM Sans', sans-serif`;
                ctx.fillText(`${Math.round(blPct)}%`, cx, bottomY + 2);
            }
        }

        ctx.restore();
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
        if (this.data.x) {
            for (const resources of Object.values(this.data.x)) {
                for (const t of Object.values(resources)) {
                    if (t > 0) total += t;
                }
            }
        }
        return total;
    }

    hitTest(mx, my) {
        // Extend hit area above rim for curtailed balls
        const topY = Math.min(this.y, this.rimY - 30);
        if (mx < this.x || mx > this.x + this.width ||
            my < topY || my > this.y + this.height) return null;
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
        this.gridBaseline = null;
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
        this.gridBaseline = deploymentData.gridBaseline || {};
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
     * Build cross-ISO flow map: for each source ISO, which buyer ISOs claim its resources.
     * Returns {sourceIso: {buyerIso: {resource: twh}}}
     */
    _buildCrossIsoFlows(strategyId) {
        if (!this.data || !this.data.data[strategyId]) return {};

        const flows = {};  // sourceIso → {buyerIso → {resource → twh}}
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

            // record.x = {sourceIso: {resource: twh}} — what buyerIso buys from sourceIso
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

        // Pre-compute cross-ISO flows per strategy for glow semantics
        const crossFlowsByStrategy = {};
        for (const strat of STRATEGIES) {
            crossFlowsByStrategy[strat] = this._buildCrossIsoFlows(strat);
        }

        // Update each jar
        for (const jar of this.jars) {
            const stratData = this.data.data[jar.strategy];
            let record = null;

            if (stratData) {
                const isoData = stratData[jar.iso];
                if (isoData) {
                    const pk = this._findClosestKey(Object.keys(isoData), this.participation);
                    if (pk) {
                        const tk = this._findClosestKey(Object.keys(isoData[pk]), this.threshold);
                        if (tk) {
                            record = isoData[pk][tk];
                        }
                    }
                }
            }

            const ballTwh = (GRID_DEMANDS[jar.iso] || 300) * 0.01;  // 1% of grid
            const bl = this.gridBaseline[jar.iso] || null;

            // Cross-ISO flows for THIS jar's ISO as source
            const crossFlows = crossFlowsByStrategy[jar.strategy][jar.iso] || null;

            jar.setBalls(record, ballTwh, bl, crossFlows);
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
        if (!this.tooltipEl) return;

        let html = `<div class="deployment-tooltip-header">
            <strong>${STRATEGY_LABELS[jar.strategy].replace('\n', ' ')}</strong> —
            <span style="color:${getIsoColor(jar.iso)}">${jar.iso}</span>
        </div>`;

        // Grid baseline info
        const bl = this.gridBaseline[jar.iso];
        if (bl) {
            html += `<div class="deployment-tooltip-stats">
                <span>Grid baseline: ${bl.totalPct.toFixed(0)}% clean</span>
            </div>`;
        }

        if (jar.data) {
            const data = jar.data;

            // Dispatch metrics
            html += '<div class="deployment-tooltip-stats">';
            if (data.hms != null) html += `<span>Hourly match: ${data.hms}%</span>`;
            if (data.gridCleanPct != null) html += `<span>Grid clean: ${data.gridCleanPct}%</span>`;
            if (data.gasGw != null) html += `<span>Gas backup: ${data.gasGw} GW</span>`;
            if (data.curtTwh != null && data.curtTwh > 0) html += `<span>Curtailed: ${data.curtTwh} TWh</span>`;
            html += '</div>';

            // Resource breakdown
            html += '<div class="deployment-tooltip-breakdown">';

            if (data.e) {
                const entries = Object.entries(data.e).filter(([, t]) => t > 0);
                if (entries.length > 0) {
                    html += '<div class="tooltip-section-label">Existing Claimed</div>';
                    for (const [res, twh] of entries) {
                        html += `<div class="tooltip-resource">
                            <span class="tooltip-dot" style="background:${getResourceColor(res)};opacity:0.4;border:2px solid ${getResourceColor(res)}"></span>
                            <span>${getResourceLabel(res)}</span>
                            <span class="tooltip-twh">${twh.toFixed(1)} TWh</span>
                        </div>`;
                    }
                }
            }

            if (data.n) {
                const entries = Object.entries(data.n).filter(([, t]) => t > 0);
                if (entries.length > 0) {
                    html += '<div class="tooltip-section-label">New Build</div>';
                    for (const [res, twh] of entries) {
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

            if (data.co2 > 0) {
                html += `<div class="deployment-tooltip-footer">
                    CO₂ reduced: ${data.co2.toFixed(1)} MtCO₂
                    ${data.co2r ? `(${data.co2r}% of baseline)` : ''}
                </div>`;
            }
        }

        // Hovered ball detail
        if (ball && ball.opacity > 0.5) {
            const tierLabel = { baseline: 'grid baseline', claimed: 'existing claimed', new: 'new build', curtailed: 'curtailed' };
            html += `<div class="deployment-tooltip-footer" style="margin-top:4px;font-style:italic">
                ${getResourceLabel(ball.resource)} (${tierLabel[ball.tier] || ball.tier})${ball.glowIso ? ` → serving ${ball.glowIso}` : ''}
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
                if (d.x) {
                    for (const [srcIso, resources] of Object.entries(d.x)) {
                        for (const [r, t] of Object.entries(resources)) {
                            if (t > 0) {
                                totalCross += t;
                                resourceBreakdown.new[r] = (resourceBreakdown.new[r] || 0) + t;
                            }
                        }
                    }
                }
                totalCO2 += d.co2 || 0;
                totalBaseline += d.bl || 0;
            }

            const totalPaid = totalExisting + totalNew + totalCross;

            stats[strat] = {
                existingTwh: totalExisting,
                newTwh: totalNew,
                crossTwh: totalCross,
                totalTwh: totalPaid,
                totalCO2Mt: totalCO2,
                totalRealCO2Mt: totalCO2,
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
