// ============================================================================
// CANVAS BANNERS — 13 animated <canvas> hero banner variants
// ============================================================================
// Loaded on pages that need canvas-based banners (e.g., about.html).
// Registers variants into window._canvasBanners for shared-header.js to use.
//
// Each banner: init(canvas, header) starts animation, destroy() stops it.
// Uses requestAnimationFrame exclusively (no setInterval).
// DPR-aware, mobile-responsive, IntersectionObserver-paused when offscreen.
// ============================================================================

(function() {
    'use strict';

    window._canvasBanners = {};

    // ---- Shared utilities ----

    var isMobile = window.innerWidth < 768;

    // Seeded PRNG (deterministic for consistent layouts)
    function makeRng(seed) {
        var s = seed;
        return function() {
            s = (s * 16807) % 2147483647;
            return s / 2147483647;
        };
    }

    // Setup canvas with DPR scaling, returns {ctx, w, h, dpr}
    function setupCanvas(canvas, header) {
        var dpr = window.devicePixelRatio || 1;
        var rect = header.getBoundingClientRect();
        var w = rect.width;
        var h = rect.height;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        canvas.style.width = w + 'px';
        canvas.style.height = h + 'px';
        var ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        return { ctx: ctx, w: w, h: h, dpr: dpr };
    }

    // Create a banner registration helper
    function registerBanner(name, theme, factory) {
        window._canvasBanners[name] = {
            theme: theme,
            _rafId: null,
            _resizeHandler: null,
            _observer: null,
            _paused: false,
            _state: null,
            _canvas: null,
            _header: null,

            init: function(canvas, header) {
                var self = this;
                this._canvas = canvas;
                this._header = header;
                var env = setupCanvas(canvas, header);
                this._state = factory.create(env);

                // Resize handler (debounced)
                var resizeTimer = null;
                this._resizeHandler = function() {
                    clearTimeout(resizeTimer);
                    resizeTimer = setTimeout(function() {
                        if (!self._canvas) return;
                        var env2 = setupCanvas(self._canvas, self._header);
                        self._state = factory.create(env2);
                    }, 200);
                };
                window.addEventListener('resize', this._resizeHandler);

                // IntersectionObserver — pause when offscreen
                if (window.IntersectionObserver) {
                    this._observer = new IntersectionObserver(function(entries) {
                        self._paused = !entries[0].isIntersecting;
                    }, { threshold: 0 });
                    this._observer.observe(header);
                }

                // RAF loop
                var tick = function(ts) {
                    if (!self._paused && self._state) {
                        factory.draw(self._state, ts);
                    }
                    self._rafId = requestAnimationFrame(tick);
                };
                this._rafId = requestAnimationFrame(tick);
            },

            destroy: function() {
                if (this._rafId) cancelAnimationFrame(this._rafId);
                this._rafId = null;
                if (this._resizeHandler) window.removeEventListener('resize', this._resizeHandler);
                this._resizeHandler = null;
                if (this._observer) { this._observer.disconnect(); this._observer = null; }
                this._state = null;
                this._canvas = null;
                this._header = null;
            }
        };
    }

    // ---- Pastel palette (used across multiple banners) ----
    var PASTELS = {
        lavender:  { r: 167, g: 139, b: 250 },
        sage:      { r: 74,  g: 222, b: 128 },
        coral:     { r: 251, g: 146, b: 131 },
        gold:      { r: 251, g: 191, b: 36  },
        blue:      { r: 96,  g: 165, b: 250 },
        teal:      { r: 45,  g: 212, b: 191 },
        rose:      { r: 244, g: 114, b: 182 }
    };
    var PASTEL_KEYS = Object.keys(PASTELS);

    function rgba(c, a) {
        return 'rgba(' + c.r + ',' + c.g + ',' + c.b + ',' + a + ')';
    }

    // Resource-aligned colors for energy-themed banners
    var RES = {
        nuclear: { r: 99, g: 102, b: 241 },   // #6366F1
        solar:   { r: 245, g: 158, b: 11 },    // #F59E0B
        wind:    { r: 34,  g: 197, b: 94 },     // #22C55E
        storage: { r: 239, g: 68,  b: 68 },     // #EF4444
        hydro:   { r: 14,  g: 165, b: 233 },    // #0EA5E9
        battery: { r: 6,   g: 182, b: 212 },    // #06B6D4
        ldes:    { r: 233, g: 30,  b: 99 }      // #E91E63
    };


    // ========================================================================
    // 1. FROSTED CIRCUIT (light bg #F8F8F7)
    // ========================================================================
    registerBanner('frosted-circuit', 'light', {
        create: function(env) {
            var rng = makeRng(42);
            var w = env.w, h = env.h;
            var nodeCount = isMobile ? 30 : 55;
            // Generate circuit nodes
            var nodes = [];
            for (var i = 0; i < nodeCount; i++) {
                var col = PASTELS[PASTEL_KEYS[Math.floor(rng() * PASTEL_KEYS.length)]];
                nodes.push({
                    x: rng() * w, y: rng() * h,
                    r: 2 + rng() * 5,
                    color: col,
                    pulse: rng() < 0.15, // ~15% pulse
                    pulsePhase: rng() * Math.PI * 2,
                    pulseSpeed: 0.8 + rng() * 0.6
                });
            }
            // Generate circuit paths (horizontal/vertical segments connecting nearby nodes)
            var paths = [];
            for (var a = 0; a < nodes.length; a++) {
                // Connect to 1-2 nearest nodes
                var best = -1, bestDist = Infinity;
                for (var b = a + 1; b < nodes.length; b++) {
                    var dx = nodes[b].x - nodes[a].x;
                    var dy = nodes[b].y - nodes[a].y;
                    var d = Math.sqrt(dx * dx + dy * dy);
                    if (d < bestDist && d < w * 0.3) {
                        bestDist = d; best = b;
                    }
                }
                if (best >= 0) {
                    var col2 = PASTELS[PASTEL_KEYS[Math.floor(rng() * PASTEL_KEYS.length)]];
                    paths.push({ a: a, b: best, color: col2 });
                }
            }
            // Sine waves
            var waves = [
                { freq: 0.003, amp: h * 0.08, yBase: h * 0.35, color: PASTELS.gold, phase: 0 },
                { freq: 0.005, amp: h * 0.06, yBase: h * 0.5, color: PASTELS.sage, phase: 1 },
                { freq: 0.004, amp: h * 0.07, yBase: h * 0.65, color: PASTELS.blue, phase: 2 },
                { freq: 0.006, amp: h * 0.05, yBase: h * 0.45, color: PASTELS.lavender, phase: 3 }
            ];
            return { ctx: env.ctx, w: w, h: h, nodes: nodes, paths: paths, waves: waves };
        },
        draw: function(st, ts) {
            var ctx = st.ctx, w = st.w, h = st.h;
            var t = ts / 1000;
            ctx.clearRect(0, 0, w, h);

            // Draw circuit paths (orthogonal with rounded corners)
            for (var p = 0; p < st.paths.length; p++) {
                var path = st.paths[p];
                var na = st.nodes[path.a], nb = st.nodes[path.b];
                ctx.beginPath();
                ctx.moveTo(na.x, na.y);
                // Go horizontal first, then vertical
                var midX = (na.x + nb.x) / 2;
                ctx.lineTo(midX, na.y);
                ctx.lineTo(midX, nb.y);
                ctx.lineTo(nb.x, nb.y);
                ctx.strokeStyle = rgba(path.color, 0.08);
                ctx.lineWidth = 1;
                ctx.stroke();
            }

            // Draw sine waves with area fill
            for (var wi = 0; wi < st.waves.length; wi++) {
                var wave = st.waves[wi];
                var phaseShift = t * 0.15 + wave.phase;
                ctx.beginPath();
                ctx.moveTo(0, h);
                for (var x = 0; x <= w; x += 3) {
                    var y = wave.yBase + Math.sin(x * wave.freq + phaseShift) * wave.amp;
                    ctx.lineTo(x, y);
                }
                ctx.lineTo(w, h);
                ctx.closePath();
                ctx.fillStyle = rgba(wave.color, 0.07);
                ctx.fill();
                // Stroke the top edge
                ctx.beginPath();
                for (var x2 = 0; x2 <= w; x2 += 3) {
                    var y2 = wave.yBase + Math.sin(x2 * wave.freq + phaseShift) * wave.amp;
                    if (x2 === 0) ctx.moveTo(x2, y2);
                    else ctx.lineTo(x2, y2);
                }
                ctx.strokeStyle = rgba(wave.color, 0.18);
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }

            // Draw nodes
            for (var ni = 0; ni < st.nodes.length; ni++) {
                var n = st.nodes[ni];
                var op = 0.35;
                if (n.pulse) {
                    op = 0.25 + 0.15 * Math.sin(t * n.pulseSpeed + n.pulsePhase);
                }
                ctx.beginPath();
                ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
                ctx.fillStyle = rgba(n.color, op);
                ctx.fill();
            }
        }
    });


    // ========================================================================
    // 2. PARTICLE FILL DARK (dark bg #0B1120)
    // ========================================================================
    registerBanner('particle-fill-dark', 'dark', {
        create: function(env) {
            var rng = makeRng(99);
            var w = env.w, h = env.h;
            // Define wave shapes (area chart silhouettes)
            var shapes = [
                { yFunc: function(x, w) { return h * 0.85 - Math.sin(x / w * Math.PI) * h * 0.15; }, color: PASTELS.teal },
                { yFunc: function(x, w) { return h * 0.75 - Math.sin(x / w * Math.PI * 1.3 + 0.5) * h * 0.2; }, color: PASTELS.coral },
                { yFunc: function(x, w) { return h * 0.65 - Math.sin(x / w * Math.PI * 0.8 + 1) * h * 0.18; }, color: PASTELS.gold },
                { yFunc: function(x, w) { return h * 0.55 - Math.sin(x / w * Math.PI * 1.5 + 2) * h * 0.12; }, color: PASTELS.lavender }
            ];
            var particleCount = isMobile ? 400 : 900;
            var particles = [];
            for (var i = 0; i < particleCount; i++) {
                var si = Math.floor(rng() * shapes.length);
                var shape = shapes[si];
                var px = rng() * w;
                var topY = shape.yFunc(px, w);
                var targetY = topY + rng() * (h - topY);
                particles.push({
                    x: px + (rng() - 0.5) * 4,
                    y: -rng() * h * 0.5, // start above
                    targetY: targetY,
                    r: 1.5 + rng() * 2.5,
                    color: shape.color,
                    speed: 0.5 + rng() * 1.5,
                    settled: false,
                    bounce: 0,
                    delay: rng() * 8 // stagger over 8 seconds
                });
            }
            return { ctx: env.ctx, w: w, h: h, particles: particles, shapes: shapes, startTime: -1 };
        },
        draw: function(st, ts) {
            var ctx = st.ctx, w = st.w, h = st.h;
            if (st.startTime < 0) st.startTime = ts;
            var elapsed = (ts - st.startTime) / 1000;
            ctx.clearRect(0, 0, w, h);

            // Draw faint shape outlines
            for (var si = 0; si < st.shapes.length; si++) {
                var shape = st.shapes[si];
                ctx.beginPath();
                for (var x = 0; x <= w; x += 4) {
                    var y = shape.yFunc(x, w);
                    if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }
                ctx.strokeStyle = rgba(shape.color, 0.1);
                ctx.lineWidth = 0.5;
                ctx.stroke();
            }

            // Draw particles
            for (var pi = 0; pi < st.particles.length; pi++) {
                var p = st.particles[pi];
                if (elapsed < p.delay) continue;
                var age = elapsed - p.delay;

                if (!p.settled) {
                    p.y += p.speed * 2;
                    if (p.y >= p.targetY) {
                        p.y = p.targetY;
                        p.settled = true;
                        p.bounce = 3;
                    }
                } else if (p.bounce > 0) {
                    p.bounce *= 0.9;
                    p.y = p.targetY - Math.abs(Math.sin(age * 8)) * p.bounce;
                }

                // Glow effect
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r + 1, 0, Math.PI * 2);
                ctx.fillStyle = rgba(p.color, 0.08);
                ctx.fill();
                // Core
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                ctx.fillStyle = rgba(p.color, p.settled ? 0.6 : 0.4);
                ctx.fill();
            }
        }
    });


    // ========================================================================
    // 3. PARTICLE FILL LIGHT (light bg)
    // ========================================================================
    registerBanner('particle-fill-light', 'light', {
        create: function(env) {
            var rng = makeRng(99);
            var w = env.w, h = env.h;
            var shapes = [
                { yFunc: function(x, w) { return h * 0.85 - Math.sin(x / w * Math.PI) * h * 0.15; }, color: RES.hydro },
                { yFunc: function(x, w) { return h * 0.75 - Math.sin(x / w * Math.PI * 1.3 + 0.5) * h * 0.2; }, color: RES.solar },
                { yFunc: function(x, w) { return h * 0.65 - Math.sin(x / w * Math.PI * 0.8 + 1) * h * 0.18; }, color: RES.wind },
                { yFunc: function(x, w) { return h * 0.55 - Math.sin(x / w * Math.PI * 1.5 + 2) * h * 0.12; }, color: RES.nuclear }
            ];
            var particleCount = isMobile ? 400 : 900;
            var particles = [];
            for (var i = 0; i < particleCount; i++) {
                var si = Math.floor(rng() * shapes.length);
                var shape = shapes[si];
                var px = rng() * w;
                var topY = shape.yFunc(px, w);
                var targetY = topY + rng() * (h - topY);
                particles.push({
                    x: px + (rng() - 0.5) * 4,
                    y: -rng() * h * 0.5,
                    targetY: targetY,
                    r: 1.5 + rng() * 2.5,
                    color: shape.color,
                    speed: 0.5 + rng() * 1.5,
                    settled: false,
                    bounce: 0,
                    delay: rng() * 8
                });
            }
            return { ctx: env.ctx, w: w, h: h, particles: particles, shapes: shapes, startTime: -1 };
        },
        draw: function(st, ts) {
            var ctx = st.ctx, w = st.w, h = st.h;
            if (st.startTime < 0) st.startTime = ts;
            var elapsed = (ts - st.startTime) / 1000;
            ctx.clearRect(0, 0, w, h);
            for (var si = 0; si < st.shapes.length; si++) {
                var shape = st.shapes[si];
                ctx.beginPath();
                for (var x = 0; x <= w; x += 4) {
                    var y = shape.yFunc(x, w);
                    if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }
                ctx.strokeStyle = rgba(shape.color, 0.15);
                ctx.lineWidth = 0.5;
                ctx.stroke();
            }
            for (var pi = 0; pi < st.particles.length; pi++) {
                var p = st.particles[pi];
                if (elapsed < p.delay) continue;
                var age = elapsed - p.delay;
                if (!p.settled) {
                    p.y += p.speed * 2;
                    if (p.y >= p.targetY) { p.y = p.targetY; p.settled = true; p.bounce = 3; }
                } else if (p.bounce > 0) {
                    p.bounce *= 0.9;
                    p.y = p.targetY - Math.abs(Math.sin(age * 8)) * p.bounce;
                }
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                ctx.fillStyle = rgba(p.color, p.settled ? 0.45 : 0.3);
                ctx.fill();
            }
        }
    });


    // ========================================================================
    // 4. DENSE CIRCUIT MAP (light bg #F5F5F3)
    // ========================================================================
    registerBanner('dense-circuit-map', 'light', {
        create: function(env) {
            var rng = makeRng(137);
            var w = env.w, h = env.h;
            var pathCount = isMobile ? 50 : 110;
            // Generate circuit paths
            var circuitPaths = [];
            for (var i = 0; i < pathCount; i++) {
                var points = [];
                var px = -20 + rng() * 40;
                var py = rng() * h;
                points.push({ x: px, y: py });
                var segCount = 4 + Math.floor(rng() * 6);
                for (var s = 0; s < segCount; s++) {
                    if (s % 2 === 0) {
                        // Horizontal segment
                        px += 40 + rng() * (w / 4);
                    } else {
                        // Vertical jog
                        py += (rng() - 0.5) * h * 0.4;
                        py = Math.max(10, Math.min(h - 10, py));
                    }
                    points.push({ x: px, y: py });
                    if (px > w + 20) break;
                }
                var col = PASTELS[PASTEL_KEYS[Math.floor(rng() * PASTEL_KEYS.length)]];
                circuitPaths.push({ points: points, color: col });
            }
            // Nodes at path endpoints and intersections
            var nodeArr = [];
            for (var ci = 0; ci < circuitPaths.length; ci++) {
                var pts = circuitPaths[ci].points;
                for (var j = 0; j < pts.length; j++) {
                    if (rng() < 0.35) {
                        var shape = rng() < 0.6 ? 'circle' : (rng() < 0.5 ? 'square' : 'ring');
                        var nc = PASTELS[PASTEL_KEYS[Math.floor(rng() * PASTEL_KEYS.length)]];
                        nodeArr.push({ x: pts[j].x, y: pts[j].y, r: 2 + rng() * 3, shape: shape, color: nc });
                    }
                }
            }
            // Traveling signal dots
            var signalCount = isMobile ? 8 : 18;
            var signals = [];
            for (var si = 0; si < signalCount; si++) {
                var pathIdx = Math.floor(rng() * circuitPaths.length);
                signals.push({
                    pathIdx: pathIdx,
                    progress: rng(), // 0-1 along path
                    speed: 0.05 + rng() * 0.15, // per second
                    color: circuitPaths[pathIdx].color
                });
            }
            return { ctx: env.ctx, w: w, h: h, paths: circuitPaths, nodes: nodeArr, signals: signals };
        },
        draw: function(st, ts) {
            var ctx = st.ctx, w = st.w, h = st.h;
            var t = ts / 1000;
            ctx.clearRect(0, 0, w, h);

            // Draw paths
            for (var pi = 0; pi < st.paths.length; pi++) {
                var path = st.paths[pi];
                ctx.beginPath();
                for (var j = 0; j < path.points.length; j++) {
                    var pt = path.points[j];
                    if (j === 0) ctx.moveTo(pt.x, pt.y);
                    else ctx.lineTo(pt.x, pt.y);
                }
                ctx.strokeStyle = rgba(path.color, 0.12);
                ctx.lineWidth = 0.8;
                ctx.lineJoin = 'round';
                ctx.stroke();
            }

            // Draw nodes
            for (var ni = 0; ni < st.nodes.length; ni++) {
                var n = st.nodes[ni];
                if (n.x < -10 || n.x > w + 10) continue;
                ctx.fillStyle = rgba(n.color, 0.35);
                if (n.shape === 'circle') {
                    ctx.beginPath();
                    ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
                    ctx.fill();
                } else if (n.shape === 'square') {
                    ctx.fillRect(n.x - n.r * 0.7, n.y - n.r * 0.7, n.r * 1.4, n.r * 1.4);
                } else { // ring
                    ctx.beginPath();
                    ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
                    ctx.strokeStyle = rgba(n.color, 0.4);
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            }

            // Draw traveling signals
            for (var si = 0; si < st.signals.length; si++) {
                var sig = st.signals[si];
                sig.progress += sig.speed / 60; // approximate per-frame
                if (sig.progress > 1) {
                    sig.progress = 0;
                    sig.pathIdx = Math.floor(Math.random() * st.paths.length);
                    sig.color = st.paths[sig.pathIdx].color;
                }
                // Interpolate position along path
                var path2 = st.paths[sig.pathIdx];
                var totalLen = path2.points.length - 1;
                var segIdx = Math.floor(sig.progress * totalLen);
                var segFrac = (sig.progress * totalLen) - segIdx;
                if (segIdx >= totalLen) segIdx = totalLen - 1;
                var pa = path2.points[segIdx];
                var pb = path2.points[Math.min(segIdx + 1, totalLen)];
                var sx = pa.x + (pb.x - pa.x) * segFrac;
                var sy = pa.y + (pb.y - pa.y) * segFrac;
                // Fade at edges
                var fade = Math.min(sig.progress * 5, (1 - sig.progress) * 5, 1);
                // Glow
                ctx.beginPath();
                ctx.arc(sx, sy, 4, 0, Math.PI * 2);
                ctx.fillStyle = rgba(sig.color, 0.15 * fade);
                ctx.fill();
                // Core dot
                ctx.beginPath();
                ctx.arc(sx, sy, 2, 0, Math.PI * 2);
                ctx.fillStyle = rgba(sig.color, 0.7 * fade);
                ctx.fill();
            }
        }
    });


    // ========================================================================
    // 5. HEATMAP PULSE DARK (dark bg #0B1120)
    // ========================================================================
    registerBanner('heatmap-pulse-dark', 'dark', {
        create: function(env) {
            var w = env.w, h = env.h;
            var cols = 24;
            var rows = isMobile ? 30 : 50;
            var cellW = w / cols;
            var cellH = h / rows;
            return { ctx: env.ctx, w: w, h: h, cols: cols, rows: rows, cellW: cellW, cellH: cellH };
        },
        draw: function(st, ts) {
            var ctx = st.ctx, w = st.w, h = st.h;
            var t = ts / 1000;
            var period = 15; // 15 second loop
            var phase = (t % period) / period; // 0-1
            ctx.clearRect(0, 0, w, h);

            // Colors: navy -> teal -> gold -> white
            for (var row = 0; row < st.rows; row++) {
                for (var col = 0; col < st.cols; col++) {
                    // Solar curve: bright during day hours (col 6-18), seasonal variation (row)
                    var hourFactor = Math.max(0, Math.sin((col - 2) / 24 * Math.PI * 2 - 0.3));
                    hourFactor = hourFactor * hourFactor; // sharpen
                    // Seasonal: summer (middle rows) brighter
                    var seasonFactor = 0.6 + 0.4 * Math.sin(row / st.rows * Math.PI);
                    // Base brightness
                    var brightness = hourFactor * seasonFactor;
                    // Animate wave sweeping across
                    var waveX = (phase * 2 - 0.5); // -0.5 to 1.5
                    var dist = Math.abs((row / st.rows) - waveX);
                    var waveBrightness = Math.max(0, 1 - dist * 3);
                    brightness = Math.max(brightness * 0.3, Math.min(1, brightness * 0.3 + waveBrightness * brightness * 0.7 + waveBrightness * 0.1));

                    // Color mapping
                    var r, g, b;
                    if (brightness < 0.33) {
                        var t1 = brightness / 0.33;
                        r = 11 + t1 * 34; g = 17 + t1 * 183; b = 32 + t1 * 159; // navy -> teal
                    } else if (brightness < 0.66) {
                        var t2 = (brightness - 0.33) / 0.33;
                        r = 45 + t2 * 206; g = 200 - t2 * 42; b = 191 - t2 * 155; // teal -> gold
                    } else {
                        var t3 = (brightness - 0.66) / 0.34;
                        r = 251 + t3 * 4; g = 158 + t3 * 97; b = 36 + t3 * 219; // gold -> white
                    }

                    ctx.fillStyle = 'rgb(' + Math.round(r) + ',' + Math.round(g) + ',' + Math.round(b) + ')';
                    ctx.fillRect(col * st.cellW, row * st.cellH, st.cellW - 0.5, st.cellH - 0.5);

                    // Subtle glow on bright cells
                    if (brightness > 0.7) {
                        ctx.fillStyle = 'rgba(255,255,255,' + ((brightness - 0.7) * 0.3) + ')';
                        ctx.fillRect(col * st.cellW, row * st.cellH, st.cellW - 0.5, st.cellH - 0.5);
                    }
                }
            }

            // Grid lines
            ctx.strokeStyle = 'rgba(255,255,255,0.03)';
            ctx.lineWidth = 0.5;
            for (var c = 0; c <= st.cols; c++) {
                ctx.beginPath();
                ctx.moveTo(c * st.cellW, 0);
                ctx.lineTo(c * st.cellW, h);
                ctx.stroke();
            }
        }
    });


    // ========================================================================
    // 6. HEATMAP PULSE LIGHT (light bg)
    // ========================================================================
    registerBanner('heatmap-pulse-light', 'light', {
        create: function(env) {
            var w = env.w, h = env.h;
            var cols = 24;
            var rows = isMobile ? 30 : 50;
            return { ctx: env.ctx, w: w, h: h, cols: cols, rows: rows, cellW: w / cols, cellH: h / rows };
        },
        draw: function(st, ts) {
            var ctx = st.ctx, w = st.w, h = st.h;
            var t = ts / 1000;
            var period = 15;
            var phase = (t % period) / period;
            ctx.clearRect(0, 0, w, h);

            for (var row = 0; row < st.rows; row++) {
                for (var col = 0; col < st.cols; col++) {
                    var hourFactor = Math.max(0, Math.sin((col - 2) / 24 * Math.PI * 2 - 0.3));
                    hourFactor = hourFactor * hourFactor;
                    var seasonFactor = 0.6 + 0.4 * Math.sin(row / st.rows * Math.PI);
                    var brightness = hourFactor * seasonFactor;
                    var waveX = (phase * 2 - 0.5);
                    var dist = Math.abs((row / st.rows) - waveX);
                    var waveBrightness = Math.max(0, 1 - dist * 3);
                    brightness = Math.max(brightness * 0.3, Math.min(1, brightness * 0.3 + waveBrightness * brightness * 0.7 + waveBrightness * 0.1));

                    // Light palette: white (dirty) -> light blue -> teal -> deep green (clean)
                    var r, g, b;
                    if (brightness < 0.33) {
                        var t1 = brightness / 0.33;
                        r = 230 - t1 * 50; g = 232 - t1 * 20; b = 240 - t1 * 10; // light gray -> light blue
                    } else if (brightness < 0.66) {
                        var t2 = (brightness - 0.33) / 0.33;
                        r = 180 - t2 * 140; g = 212 - t2 * 15; b = 230 - t2 * 60; // light blue -> teal
                    } else {
                        var t3 = (brightness - 0.66) / 0.34;
                        r = 40 - t3 * 15; g = 197 + t3 * 30; b = 170 - t3 * 70; // teal -> bright green
                    }

                    ctx.fillStyle = 'rgb(' + Math.round(r) + ',' + Math.round(g) + ',' + Math.round(b) + ')';
                    ctx.fillRect(col * st.cellW, row * st.cellH, st.cellW - 0.5, st.cellH - 0.5);
                }
            }

            ctx.strokeStyle = 'rgba(0,0,0,0.04)';
            ctx.lineWidth = 0.5;
            for (var c = 0; c <= st.cols; c++) {
                ctx.beginPath();
                ctx.moveTo(c * st.cellW, 0);
                ctx.lineTo(c * st.cellW, h);
                ctx.stroke();
            }
        }
    });


    // ========================================================================
    // 7. STACKED CURVES DARK (dark bg #0B1120)
    // ========================================================================
    registerBanner('stacked-curves-dark', 'dark', {
        create: function(env) {
            var w = env.w, h = env.h;
            // 6 stacked layers: nuclear (flat), solar (hump), wind (irregular), storage (thin), hydro (steady), demand (top line)
            var layers = [
                { name: 'nuclear', color: RES.nuclear, baseY: 0.88, amp: 0.02, freq: 0.5, phase: 0 },
                { name: 'hydro',   color: RES.hydro,   baseY: 0.78, amp: 0.04, freq: 0.8, phase: 1 },
                { name: 'solar',   color: RES.solar,   baseY: 0.55, amp: 0.12, freq: 1.2, phase: 0.5 },
                { name: 'wind',    color: RES.wind,    baseY: 0.42, amp: 0.08, freq: 2.0, phase: 2 },
                { name: 'storage', color: RES.storage, baseY: 0.35, amp: 0.05, freq: 1.5, phase: 3 }
            ];
            return { ctx: env.ctx, w: w, h: h, layers: layers };
        },
        draw: function(st, ts) {
            var ctx = st.ctx, w = st.w, h = st.h;
            var t = ts / 1000;
            ctx.clearRect(0, 0, w, h);

            // Subtle time-axis grid
            ctx.strokeStyle = 'rgba(255,255,255,0.03)';
            ctx.lineWidth = 0.5;
            for (var gx = 0; gx < w; gx += w / 24) {
                ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
            }

            // Draw stacked areas bottom-up
            for (var li = 0; li < st.layers.length; li++) {
                var layer = st.layers[li];
                ctx.beginPath();
                ctx.moveTo(0, h);
                for (var x = 0; x <= w; x += 3) {
                    var base = layer.baseY * h;
                    var undulate = Math.sin(x / w * Math.PI * layer.freq + t * 0.3 + layer.phase) * layer.amp * h;
                    var undulate2 = Math.sin(x / w * Math.PI * layer.freq * 2.3 + t * 0.15 + layer.phase * 1.7) * layer.amp * h * 0.3;
                    var y = base + undulate + undulate2;
                    ctx.lineTo(x, y);
                }
                ctx.lineTo(w, h);
                ctx.closePath();
                // Fill with low opacity (glass effect)
                ctx.fillStyle = rgba(layer.color, 0.12);
                ctx.fill();
                // Top edge with glow
                ctx.beginPath();
                for (var x2 = 0; x2 <= w; x2 += 3) {
                    var base2 = layer.baseY * h;
                    var u = Math.sin(x2 / w * Math.PI * layer.freq + t * 0.3 + layer.phase) * layer.amp * h;
                    var u2 = Math.sin(x2 / w * Math.PI * layer.freq * 2.3 + t * 0.15 + layer.phase * 1.7) * layer.amp * h * 0.3;
                    var y2 = base2 + u + u2;
                    if (x2 === 0) ctx.moveTo(x2, y2); else ctx.lineTo(x2, y2);
                }
                ctx.strokeStyle = rgba(layer.color, 0.6);
                ctx.lineWidth = 1.2;
                ctx.stroke();
                // Outer glow
                ctx.strokeStyle = rgba(layer.color, 0.15);
                ctx.lineWidth = 4;
                ctx.stroke();
            }

            // Demand line (white, thin, on top)
            ctx.beginPath();
            for (var dx = 0; dx <= w; dx += 3) {
                var dy = h * 0.28 + Math.sin(dx / w * Math.PI * 0.7 + t * 0.2) * h * 0.03;
                if (dx === 0) ctx.moveTo(dx, dy); else ctx.lineTo(dx, dy);
            }
            ctx.strokeStyle = 'rgba(255,255,255,0.5)';
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Occasional floating particles
            var particleCount = 12;
            for (var pi = 0; pi < particleCount; pi++) {
                var px = (pi / particleCount * w + t * 15) % w;
                var py = h * (0.3 + 0.5 * Math.sin(pi * 2.7 + t * 0.2));
                var pOp = 0.15 + 0.1 * Math.sin(t * 0.5 + pi);
                ctx.beginPath();
                ctx.arc(px, py - t * 3 % 20, 1.5, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(255,255,255,' + pOp + ')';
                ctx.fill();
            }
        }
    });


    // ========================================================================
    // 8. STACKED CURVES LIGHT (light bg)
    // ========================================================================
    registerBanner('stacked-curves-light', 'light', {
        create: function(env) {
            var w = env.w, h = env.h;
            var layers = [
                { name: 'nuclear', color: RES.nuclear, baseY: 0.88, amp: 0.02, freq: 0.5, phase: 0 },
                { name: 'hydro',   color: RES.hydro,   baseY: 0.78, amp: 0.04, freq: 0.8, phase: 1 },
                { name: 'solar',   color: RES.solar,   baseY: 0.55, amp: 0.12, freq: 1.2, phase: 0.5 },
                { name: 'wind',    color: RES.wind,    baseY: 0.42, amp: 0.08, freq: 2.0, phase: 2 },
                { name: 'storage', color: RES.storage, baseY: 0.35, amp: 0.05, freq: 1.5, phase: 3 }
            ];
            return { ctx: env.ctx, w: w, h: h, layers: layers };
        },
        draw: function(st, ts) {
            var ctx = st.ctx, w = st.w, h = st.h;
            var t = ts / 1000;
            ctx.clearRect(0, 0, w, h);
            ctx.strokeStyle = 'rgba(0,0,0,0.03)';
            ctx.lineWidth = 0.5;
            for (var gx = 0; gx < w; gx += w / 24) {
                ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
            }
            for (var li = 0; li < st.layers.length; li++) {
                var layer = st.layers[li];
                ctx.beginPath();
                ctx.moveTo(0, h);
                for (var x = 0; x <= w; x += 3) {
                    var base = layer.baseY * h;
                    var u = Math.sin(x / w * Math.PI * layer.freq + t * 0.3 + layer.phase) * layer.amp * h;
                    var u2 = Math.sin(x / w * Math.PI * layer.freq * 2.3 + t * 0.15 + layer.phase * 1.7) * layer.amp * h * 0.3;
                    ctx.lineTo(x, base + u + u2);
                }
                ctx.lineTo(w, h);
                ctx.closePath();
                ctx.fillStyle = rgba(layer.color, 0.15);
                ctx.fill();
                // Top edge
                ctx.beginPath();
                for (var x2 = 0; x2 <= w; x2 += 3) {
                    var b2 = layer.baseY * h;
                    var v = Math.sin(x2 / w * Math.PI * layer.freq + t * 0.3 + layer.phase) * layer.amp * h;
                    var v2 = Math.sin(x2 / w * Math.PI * layer.freq * 2.3 + t * 0.15 + layer.phase * 1.7) * layer.amp * h * 0.3;
                    if (x2 === 0) ctx.moveTo(x2, b2 + v + v2); else ctx.lineTo(x2, b2 + v + v2);
                }
                ctx.strokeStyle = rgba(layer.color, 0.55);
                ctx.lineWidth = 1.2;
                ctx.stroke();
            }
            // Demand line
            ctx.beginPath();
            for (var dx = 0; dx <= w; dx += 3) {
                var dy = h * 0.28 + Math.sin(dx / w * Math.PI * 0.7 + t * 0.2) * h * 0.03;
                if (dx === 0) ctx.moveTo(dx, dy); else ctx.lineTo(dx, dy);
            }
            ctx.strokeStyle = 'rgba(15,26,46,0.35)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([6, 3]);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    });


    // ========================================================================
    // 9. NETWORK CONSTELLATION DARK (dark bg #0B1120)
    // ========================================================================
    registerBanner('network-constellation-dark', 'dark', {
        create: function(env) {
            var rng = makeRng(314);
            var w = env.w, h = env.h;
            var nodeCount = isMobile ? 50 : 100;
            var connectionDist = isMobile ? 100 : 150;
            // Create cluster centers (3-4 dense regions)
            var clusters = [
                { x: w * 0.2, y: h * 0.4 },
                { x: w * 0.5, y: h * 0.6 },
                { x: w * 0.75, y: h * 0.35 },
                { x: w * 0.9, y: h * 0.7 }
            ];
            var nodes = [];
            for (var i = 0; i < nodeCount; i++) {
                // 70% near clusters, 30% random
                var nearCluster = rng() < 0.7;
                var nx, ny;
                if (nearCluster) {
                    var ci = Math.floor(rng() * clusters.length);
                    nx = clusters[ci].x + (rng() - 0.5) * w * 0.25;
                    ny = clusters[ci].y + (rng() - 0.5) * h * 0.5;
                } else {
                    nx = rng() * w;
                    ny = rng() * h;
                }
                var col = PASTELS[PASTEL_KEYS[Math.floor(rng() * PASTEL_KEYS.length)]];
                nodes.push({
                    x: nx, y: ny,
                    vx: (rng() - 0.5) * 0.3,
                    vy: (rng() - 0.5) * 0.2,
                    r: 2 + rng() * 4,
                    color: col,
                    baseX: nx, baseY: ny
                });
            }
            // Pulse state
            var pulses = [];
            return { ctx: env.ctx, w: w, h: h, nodes: nodes, connectionDist: connectionDist, pulses: pulses, lastPulse: 0 };
        },
        draw: function(st, ts) {
            var ctx = st.ctx, w = st.w, h = st.h;
            var t = ts / 1000;
            ctx.clearRect(0, 0, w, h);

            // Update node positions (gentle drift)
            for (var i = 0; i < st.nodes.length; i++) {
                var n = st.nodes[i];
                n.x = n.baseX + Math.sin(t * 0.3 + i * 0.7) * 15;
                n.y = n.baseY + Math.cos(t * 0.25 + i * 1.1) * 10;
            }

            // Draw connections
            var cd2 = st.connectionDist * st.connectionDist;
            for (var a = 0; a < st.nodes.length; a++) {
                for (var b = a + 1; b < st.nodes.length; b++) {
                    var dx = st.nodes[b].x - st.nodes[a].x;
                    var dy = st.nodes[b].y - st.nodes[a].y;
                    var d2 = dx * dx + dy * dy;
                    if (d2 < cd2) {
                        var alpha = (1 - Math.sqrt(d2) / st.connectionDist) * 0.1;
                        ctx.beginPath();
                        ctx.moveTo(st.nodes[a].x, st.nodes[a].y);
                        ctx.lineTo(st.nodes[b].x, st.nodes[b].y);
                        ctx.strokeStyle = 'rgba(148,163,184,' + alpha + ')';
                        ctx.lineWidth = 0.5;
                        ctx.stroke();
                    }
                }
            }

            // Pulse effect — ripple from random node
            if (t - st.lastPulse > 3) {
                st.lastPulse = t;
                var pn = st.nodes[Math.floor(Math.random() * st.nodes.length)];
                st.pulses.push({ x: pn.x, y: pn.y, startTime: t, color: pn.color });
            }
            // Draw & cull pulses
            for (var pi = st.pulses.length - 1; pi >= 0; pi--) {
                var pulse = st.pulses[pi];
                var age = t - pulse.startTime;
                if (age > 3) { st.pulses.splice(pi, 1); continue; }
                var radius = age * 60;
                var op = Math.max(0, 0.25 - age * 0.08);
                ctx.beginPath();
                ctx.arc(pulse.x, pulse.y, radius, 0, Math.PI * 2);
                ctx.strokeStyle = rgba(pulse.color, op);
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }

            // Draw nodes
            for (var ni = 0; ni < st.nodes.length; ni++) {
                var nd = st.nodes[ni];
                // Glow
                ctx.beginPath();
                ctx.arc(nd.x, nd.y, nd.r + 2, 0, Math.PI * 2);
                ctx.fillStyle = rgba(nd.color, 0.08);
                ctx.fill();
                // Core
                ctx.beginPath();
                ctx.arc(nd.x, nd.y, nd.r, 0, Math.PI * 2);
                ctx.fillStyle = rgba(nd.color, 0.5);
                ctx.fill();
            }
        }
    });


    // ========================================================================
    // 10. NETWORK CONSTELLATION LIGHT (light bg)
    // ========================================================================
    registerBanner('network-constellation-light', 'light', {
        create: function(env) {
            var rng = makeRng(314);
            var w = env.w, h = env.h;
            var nodeCount = isMobile ? 50 : 100;
            var connectionDist = isMobile ? 100 : 150;
            var clusters = [
                { x: w * 0.2, y: h * 0.4 },
                { x: w * 0.5, y: h * 0.6 },
                { x: w * 0.75, y: h * 0.35 },
                { x: w * 0.9, y: h * 0.7 }
            ];
            var nodes = [];
            for (var i = 0; i < nodeCount; i++) {
                var nearCluster = rng() < 0.7;
                var nx, ny;
                if (nearCluster) {
                    var ci = Math.floor(rng() * clusters.length);
                    nx = clusters[ci].x + (rng() - 0.5) * w * 0.25;
                    ny = clusters[ci].y + (rng() - 0.5) * h * 0.5;
                } else {
                    nx = rng() * w;
                    ny = rng() * h;
                }
                // Use deeper resource colors for light bg
                var colors = [RES.nuclear, RES.hydro, RES.solar, RES.wind, RES.battery, RES.ldes];
                var col = colors[Math.floor(rng() * colors.length)];
                nodes.push({
                    x: nx, y: ny,
                    r: 2 + rng() * 4,
                    color: col,
                    baseX: nx, baseY: ny
                });
            }
            return { ctx: env.ctx, w: w, h: h, nodes: nodes, connectionDist: connectionDist, pulses: [], lastPulse: 0 };
        },
        draw: function(st, ts) {
            var ctx = st.ctx, w = st.w, h = st.h;
            var t = ts / 1000;
            ctx.clearRect(0, 0, w, h);
            for (var i = 0; i < st.nodes.length; i++) {
                var n = st.nodes[i];
                n.x = n.baseX + Math.sin(t * 0.3 + i * 0.7) * 15;
                n.y = n.baseY + Math.cos(t * 0.25 + i * 1.1) * 10;
            }
            var cd2 = st.connectionDist * st.connectionDist;
            for (var a = 0; a < st.nodes.length; a++) {
                for (var b2 = a + 1; b2 < st.nodes.length; b2++) {
                    var dx = st.nodes[b2].x - st.nodes[a].x;
                    var dy = st.nodes[b2].y - st.nodes[a].y;
                    var d2 = dx * dx + dy * dy;
                    if (d2 < cd2) {
                        var alpha = (1 - Math.sqrt(d2) / st.connectionDist) * 0.12;
                        ctx.beginPath();
                        ctx.moveTo(st.nodes[a].x, st.nodes[a].y);
                        ctx.lineTo(st.nodes[b2].x, st.nodes[b2].y);
                        ctx.strokeStyle = 'rgba(30,41,59,' + alpha + ')';
                        ctx.lineWidth = 0.5;
                        ctx.stroke();
                    }
                }
            }
            if (t - st.lastPulse > 3) {
                st.lastPulse = t;
                var pn = st.nodes[Math.floor(Math.random() * st.nodes.length)];
                st.pulses.push({ x: pn.x, y: pn.y, startTime: t, color: pn.color });
            }
            for (var pi = st.pulses.length - 1; pi >= 0; pi--) {
                var pulse = st.pulses[pi];
                var age = t - pulse.startTime;
                if (age > 3) { st.pulses.splice(pi, 1); continue; }
                var radius = age * 60;
                var op = Math.max(0, 0.2 - age * 0.07);
                ctx.beginPath();
                ctx.arc(pulse.x, pulse.y, radius, 0, Math.PI * 2);
                ctx.strokeStyle = rgba(pulse.color, op);
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
            for (var ni = 0; ni < st.nodes.length; ni++) {
                var nd = st.nodes[ni];
                ctx.beginPath();
                ctx.arc(nd.x, nd.y, nd.r, 0, Math.PI * 2);
                ctx.fillStyle = rgba(nd.color, 0.45);
                ctx.fill();
            }
        }
    });


    // ========================================================================
    // 11. FREQUENCY SPECTRUM (dark bg #0B1120)
    // ========================================================================
    registerBanner('frequency-spectrum', 'dark', {
        create: function(env) {
            var w = env.w, h = env.h;
            var barCount = isMobile ? 40 : 70;
            var bars = [];
            for (var i = 0; i < barCount; i++) {
                var t = i / barCount;
                // Color gradient: teal -> sage -> gold -> coral -> lavender
                var r, g, b;
                if (t < 0.25) {
                    var f = t / 0.25;
                    r = 45 + f * 29; g = 212 - f * 10; b = 191 - f * 63;
                } else if (t < 0.5) {
                    var f2 = (t - 0.25) / 0.25;
                    r = 74 + f2 * 177; g = 202 - f2 * 11; b = 128 - f2 * 92;
                } else if (t < 0.75) {
                    var f3 = (t - 0.5) / 0.25;
                    r = 251 + f3 * 0; g = 191 - f3 * 45; b = 36 + f3 * 95;
                } else {
                    var f4 = (t - 0.75) / 0.25;
                    r = 251 - f4 * 84; g = 146 - f4 * 7; b = 131 + f4 * 119;
                }
                bars.push({
                    x: (i + 0.5) / barCount * w,
                    color: { r: Math.round(r), g: Math.round(g), b: Math.round(b) },
                    freq: 0.3 + Math.random() * 1.5,
                    phase: Math.random() * Math.PI * 2,
                    maxH: 0.3 + Math.random() * 0.5
                });
            }
            return { ctx: env.ctx, w: w, h: h, bars: bars, barWidth: Math.max(2, w / barCount * 0.4) };
        },
        draw: function(st, ts) {
            var ctx = st.ctx, w = st.w, h = st.h;
            var t = ts / 1000;
            ctx.clearRect(0, 0, w, h);

            for (var i = 0; i < st.bars.length; i++) {
                var bar = st.bars[i];
                var oscillation = (Math.sin(t * bar.freq + bar.phase) + 1) / 2; // 0-1
                // Add a wave pattern across all bars
                var waveEffect = (Math.sin(t * 0.5 + i * 0.15) + 1) / 2;
                var barH = h * bar.maxH * (oscillation * 0.6 + waveEffect * 0.4);
                var barY = h - barH;

                // Glass-morphism: transparent fill + bright border + glow
                var bw = st.barWidth;

                // Glow
                ctx.fillStyle = rgba(bar.color, 0.1);
                ctx.beginPath();
                ctx.roundRect(bar.x - bw / 2 - 2, barY - 2, bw + 4, barH + 4, bw / 2 + 2);
                ctx.fill();

                // Fill (transparent)
                ctx.fillStyle = rgba(bar.color, 0.2);
                ctx.beginPath();
                ctx.roundRect(bar.x - bw / 2, barY, bw, barH, bw / 2);
                ctx.fill();

                // Border (bright)
                ctx.strokeStyle = rgba(bar.color, 0.6);
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.roundRect(bar.x - bw / 2, barY, bw, barH, bw / 2);
                ctx.stroke();
            }
        }
    });


    // ========================================================================
    // 12. TOPOGRAPHIC CONTOURS LIGHT (light bg #F5F5F3)
    // ========================================================================
    registerBanner('topographic-contours-light', 'light', {
        create: function(env) {
            var rng = makeRng(271);
            var w = env.w, h = env.h;
            var contourCount = isMobile ? 10 : 18;
            // Generate contour centers (2-3 "peaks")
            var peaks = [
                { x: w * 0.35, y: h * 0.45, spread: Math.min(w, h) * 0.7 },
                { x: w * 0.7, y: h * 0.55, spread: Math.min(w, h) * 0.5 }
            ];
            var contours = [];
            for (var i = 0; i < contourCount; i++) {
                var peakIdx = Math.floor(rng() * peaks.length);
                var peak = peaks[peakIdx];
                var level = (i + 1) / contourCount; // 0-1 (inner to outer)
                var col = PASTELS[PASTEL_KEYS[Math.floor(rng() * PASTEL_KEYS.length)]];
                // Generate control points for the contour
                var pointCount = 8 + Math.floor(rng() * 4);
                var points = [];
                for (var j = 0; j < pointCount; j++) {
                    var angle = (j / pointCount) * Math.PI * 2;
                    var radius = peak.spread * level * (0.3 + rng() * 0.4);
                    points.push({
                        x: peak.x + Math.cos(angle) * radius,
                        y: peak.y + Math.sin(angle) * radius * 0.6, // flatten vertically
                        morphSpeed: 0.1 + rng() * 0.2,
                        morphAmp: 5 + rng() * 15,
                        morphPhase: rng() * Math.PI * 2
                    });
                }
                contours.push({ points: points, color: col, level: level });
            }
            // Dots at contour intersections
            var dots = [];
            for (var di = 0; di < 30; di++) {
                var ci = Math.floor(rng() * contours.length);
                var cpt = contours[ci].points[Math.floor(rng() * contours[ci].points.length)];
                dots.push({ x: cpt.x, y: cpt.y, color: contours[ci].color, pulsePhase: rng() * Math.PI * 2 });
            }
            return { ctx: env.ctx, w: w, h: h, contours: contours, dots: dots };
        },
        draw: function(st, ts) {
            var ctx = st.ctx, w = st.w, h = st.h;
            var t = ts / 1000;
            ctx.clearRect(0, 0, w, h);

            for (var ci = 0; ci < st.contours.length; ci++) {
                var contour = st.contours[ci];
                var pts = contour.points;
                ctx.beginPath();
                // Smooth closed curve using quadratic bezier between midpoints
                var first = pts[0];
                var morphX0 = Math.sin(t * first.morphSpeed + first.morphPhase) * first.morphAmp;
                var morphY0 = Math.cos(t * first.morphSpeed * 1.3 + first.morphPhase) * first.morphAmp * 0.5;
                var startX = (first.x + morphX0 + pts[pts.length - 1].x + Math.sin(t * pts[pts.length - 1].morphSpeed + pts[pts.length - 1].morphPhase) * pts[pts.length - 1].morphAmp) / 2;
                var startY = (first.y + morphY0 + pts[pts.length - 1].y + Math.cos(t * pts[pts.length - 1].morphSpeed * 1.3 + pts[pts.length - 1].morphPhase) * pts[pts.length - 1].morphAmp * 0.5) / 2;
                ctx.moveTo(startX, startY);
                for (var pi = 0; pi < pts.length; pi++) {
                    var curr = pts[pi];
                    var next = pts[(pi + 1) % pts.length];
                    var mx = Math.sin(t * curr.morphSpeed + curr.morphPhase) * curr.morphAmp;
                    var my = Math.cos(t * curr.morphSpeed * 1.3 + curr.morphPhase) * curr.morphAmp * 0.5;
                    var nx = Math.sin(t * next.morphSpeed + next.morphPhase) * next.morphAmp;
                    var ny = Math.cos(t * next.morphSpeed * 1.3 + next.morphPhase) * next.morphAmp * 0.5;
                    var midX = ((curr.x + mx) + (next.x + nx)) / 2;
                    var midY = ((curr.y + my) + (next.y + ny)) / 2;
                    ctx.quadraticCurveTo(curr.x + mx, curr.y + my, midX, midY);
                }
                ctx.closePath();
                ctx.fillStyle = rgba(contour.color, 0.06);
                ctx.fill();
                ctx.strokeStyle = rgba(contour.color, 0.25);
                ctx.lineWidth = 1;
                ctx.stroke();
            }

            // Pulsing dots
            for (var di = 0; di < st.dots.length; di++) {
                var d = st.dots[di];
                var op = 0.2 + 0.15 * Math.sin(t * 0.8 + d.pulsePhase);
                ctx.beginPath();
                ctx.arc(d.x, d.y, 2.5, 0, Math.PI * 2);
                ctx.fillStyle = rgba(d.color, op);
                ctx.fill();
            }
        }
    });


    // ========================================================================
    // 13. TOPOGRAPHIC CONTOURS DARK (dark bg #0B1120)
    // ========================================================================
    registerBanner('topographic-contours-dark', 'dark', {
        create: function(env) {
            var rng = makeRng(271);
            var w = env.w, h = env.h;
            var contourCount = isMobile ? 10 : 18;
            var peaks = [
                { x: w * 0.35, y: h * 0.45, spread: Math.min(w, h) * 0.7 },
                { x: w * 0.7, y: h * 0.55, spread: Math.min(w, h) * 0.5 }
            ];
            var contours = [];
            for (var i = 0; i < contourCount; i++) {
                var peakIdx = Math.floor(rng() * peaks.length);
                var peak = peaks[peakIdx];
                var level = (i + 1) / contourCount;
                var col = PASTELS[PASTEL_KEYS[Math.floor(rng() * PASTEL_KEYS.length)]];
                var pointCount = 8 + Math.floor(rng() * 4);
                var points = [];
                for (var j = 0; j < pointCount; j++) {
                    var angle = (j / pointCount) * Math.PI * 2;
                    var radius = peak.spread * level * (0.3 + rng() * 0.4);
                    points.push({
                        x: peak.x + Math.cos(angle) * radius,
                        y: peak.y + Math.sin(angle) * radius * 0.6,
                        morphSpeed: 0.1 + rng() * 0.2,
                        morphAmp: 5 + rng() * 15,
                        morphPhase: rng() * Math.PI * 2
                    });
                }
                contours.push({ points: points, color: col, level: level });
            }
            var dots = [];
            for (var di = 0; di < 30; di++) {
                var ci = Math.floor(rng() * contours.length);
                var cpt = contours[ci].points[Math.floor(rng() * contours[ci].points.length)];
                dots.push({ x: cpt.x, y: cpt.y, color: contours[ci].color, pulsePhase: rng() * Math.PI * 2 });
            }
            return { ctx: env.ctx, w: w, h: h, contours: contours, dots: dots };
        },
        draw: function(st, ts) {
            var ctx = st.ctx, w = st.w, h = st.h;
            var t = ts / 1000;
            ctx.clearRect(0, 0, w, h);
            for (var ci = 0; ci < st.contours.length; ci++) {
                var contour = st.contours[ci];
                var pts = contour.points;
                ctx.beginPath();
                var first = pts[0];
                var mx0 = Math.sin(t * first.morphSpeed + first.morphPhase) * first.morphAmp;
                var my0 = Math.cos(t * first.morphSpeed * 1.3 + first.morphPhase) * first.morphAmp * 0.5;
                var last = pts[pts.length - 1];
                var startX = (first.x + mx0 + last.x + Math.sin(t * last.morphSpeed + last.morphPhase) * last.morphAmp) / 2;
                var startY = (first.y + my0 + last.y + Math.cos(t * last.morphSpeed * 1.3 + last.morphPhase) * last.morphAmp * 0.5) / 2;
                ctx.moveTo(startX, startY);
                for (var pi = 0; pi < pts.length; pi++) {
                    var curr = pts[pi];
                    var next = pts[(pi + 1) % pts.length];
                    var mx = Math.sin(t * curr.morphSpeed + curr.morphPhase) * curr.morphAmp;
                    var my = Math.cos(t * curr.morphSpeed * 1.3 + curr.morphPhase) * curr.morphAmp * 0.5;
                    var nx = Math.sin(t * next.morphSpeed + next.morphPhase) * next.morphAmp;
                    var ny = Math.cos(t * next.morphSpeed * 1.3 + next.morphPhase) * next.morphAmp * 0.5;
                    var midX = ((curr.x + mx) + (next.x + nx)) / 2;
                    var midY = ((curr.y + my) + (next.y + ny)) / 2;
                    ctx.quadraticCurveTo(curr.x + mx, curr.y + my, midX, midY);
                }
                ctx.closePath();
                ctx.fillStyle = rgba(contour.color, 0.05);
                ctx.fill();
                ctx.strokeStyle = rgba(contour.color, 0.3);
                ctx.lineWidth = 1;
                ctx.stroke();
            }
            for (var di = 0; di < st.dots.length; di++) {
                var d = st.dots[di];
                var op = 0.2 + 0.15 * Math.sin(t * 0.8 + d.pulsePhase);
                ctx.beginPath();
                ctx.arc(d.x, d.y, 2.5, 0, Math.PI * 2);
                ctx.fillStyle = rgba(d.color, op);
                ctx.fill();
            }
        }
    });

})();
