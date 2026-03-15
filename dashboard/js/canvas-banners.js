// ============================================================================
// CANVAS BANNERS — 10 animated <canvas> hero banner variants
// ============================================================================
// Loaded on pages that need canvas-based banners (e.g., about.html).
// Registers variants into window._canvasBanners for shared-header.js to use.
//
// Variants:
//   dense-circuit-map (light), dense-circuit-white (light), dense-circuit-navy (dark)
//   frequency-spectrum (dark), frequency-light (light)
//   stacked-curves-dark (dark)
//   network-constellation-light (light)
//   frequency-dense-navy (dark), frequency-dense-white (light) [composites]
//   stacked-dense-navy (dark), stacked-dense-white (light) [composites]
//
// Each banner: init(canvas, header) starts animation, destroy() stops it.
// Uses requestAnimationFrame exclusively (no setInterval).
// DPR-aware, mobile-responsive, IntersectionObserver-paused when offscreen.
// ============================================================================

(function() {
    'use strict';

    window._canvasBanners = {};

    // ---- roundRect polyfill for older browsers ----
    if (typeof CanvasRenderingContext2D !== 'undefined' &&
        !CanvasRenderingContext2D.prototype.roundRect) {
        CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {
            if (typeof r === 'number') r = [r];
            var rad = r[0] || 0;
            this.moveTo(x + rad, y);
            this.lineTo(x + w - rad, y);
            this.arcTo(x + w, y, x + w, y + rad, rad);
            this.lineTo(x + w, y + h - rad);
            this.arcTo(x + w, y + h, x + w - rad, y + h, rad);
            this.lineTo(x + rad, y + h);
            this.arcTo(x, y + h, x, y + h - rad, rad);
            this.lineTo(x, y + rad);
            this.arcTo(x, y, x + rad, y, rad);
            return this;
        };
    }

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

                // RAF loop (try/catch prevents one bad frame from killing animation)
                var errCount = 0;
                var tick = function(ts) {
                    if (!self._paused && self._state) {
                        try {
                            factory.draw(self._state, ts);
                        } catch (e) {
                            if (++errCount <= 3) console.warn('Banner draw error:', e);
                        }
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


    // ====================================================================
    // REUSABLE LAYER FUNCTIONS
    // ====================================================================
    // These are extracted from standalone banners so composite banners
    // can layer multiple animations (e.g., circuit + frequency).
    // Each pair: createXxxState(env, opts) → state, drawXxx(ctx, w, h, state, t, opts)

    // ---- Dense Circuit layer ----

    function createCircuitState(env, opts) {
        opts = opts || {};
        var rng = makeRng(opts.seed || 137);
        var w = env.w, h = env.h;
        var pathCount = isMobile ? 50 : 110;
        var circuitPaths = [];
        for (var i = 0; i < pathCount; i++) {
            var points = [];
            var px = -20 + rng() * 40;
            var py = rng() * h;
            points.push({ x: px, y: py });
            var segCount = 4 + Math.floor(rng() * 6);
            for (var s = 0; s < segCount; s++) {
                if (s % 2 === 0) {
                    px += 40 + rng() * (w / 4);
                } else {
                    py += (rng() - 0.5) * h * 0.4;
                    py = Math.max(10, Math.min(h - 10, py));
                }
                points.push({ x: px, y: py });
                if (px > w + 20) break;
            }
            var col = PASTELS[PASTEL_KEYS[Math.floor(rng() * PASTEL_KEYS.length)]];
            circuitPaths.push({ points: points, color: col });
        }
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
        var signalCount = isMobile ? 8 : 18;
        var signals = [];
        for (var si = 0; si < signalCount; si++) {
            var pathIdx = Math.floor(rng() * circuitPaths.length);
            signals.push({
                pathIdx: pathIdx,
                progress: rng(),
                speed: 0.05 + rng() * 0.15,
                color: circuitPaths[pathIdx].color
            });
        }
        return { paths: circuitPaths, nodes: nodeArr, signals: signals };
    }

    function drawCircuit(ctx, w, h, cs, t, opts) {
        opts = opts || {};
        var pathOp = opts.pathOpacity !== undefined ? opts.pathOpacity : 0.12;
        var nodeOp = opts.nodeOpacity !== undefined ? opts.nodeOpacity : 0.35;
        var ringOp = opts.ringOpacity !== undefined ? opts.ringOpacity : 0.4;
        var sigGlow = opts.signalGlowOpacity !== undefined ? opts.signalGlowOpacity : 0.15;
        var sigCore = opts.signalCoreOpacity !== undefined ? opts.signalCoreOpacity : 0.7;

        // Draw paths
        for (var pi = 0; pi < cs.paths.length; pi++) {
            var path = cs.paths[pi];
            ctx.beginPath();
            for (var j = 0; j < path.points.length; j++) {
                var pt = path.points[j];
                if (j === 0) ctx.moveTo(pt.x, pt.y);
                else ctx.lineTo(pt.x, pt.y);
            }
            ctx.strokeStyle = rgba(path.color, pathOp);
            ctx.lineWidth = 0.8;
            ctx.lineJoin = 'round';
            ctx.stroke();
        }

        // Draw nodes
        for (var ni = 0; ni < cs.nodes.length; ni++) {
            var n = cs.nodes[ni];
            if (n.x < -10 || n.x > w + 10) continue;
            if (n.shape === 'circle') {
                ctx.beginPath();
                ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
                ctx.fillStyle = rgba(n.color, nodeOp);
                ctx.fill();
            } else if (n.shape === 'square') {
                ctx.fillStyle = rgba(n.color, nodeOp);
                ctx.fillRect(n.x - n.r * 0.7, n.y - n.r * 0.7, n.r * 1.4, n.r * 1.4);
            } else { // ring
                ctx.beginPath();
                ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
                ctx.strokeStyle = rgba(n.color, ringOp);
                ctx.lineWidth = 1;
                ctx.stroke();
            }
        }

        // Draw traveling signals
        for (var si = 0; si < cs.signals.length; si++) {
            var sig = cs.signals[si];
            sig.progress += sig.speed / 60;
            if (sig.progress > 1) {
                sig.progress = 0;
                sig.pathIdx = Math.floor(Math.random() * cs.paths.length);
                sig.color = cs.paths[sig.pathIdx].color;
            }
            var path2 = cs.paths[sig.pathIdx];
            var totalLen = path2.points.length - 1;
            var segIdx = Math.floor(sig.progress * totalLen);
            var segFrac = (sig.progress * totalLen) - segIdx;
            if (segIdx >= totalLen) segIdx = totalLen - 1;
            var pa = path2.points[segIdx];
            var pb = path2.points[Math.min(segIdx + 1, totalLen)];
            var sx = pa.x + (pb.x - pa.x) * segFrac;
            var sy = pa.y + (pb.y - pa.y) * segFrac;
            var fade = Math.min(sig.progress * 5, (1 - sig.progress) * 5, 1);
            // Glow
            ctx.beginPath();
            ctx.arc(sx, sy, 4, 0, Math.PI * 2);
            ctx.fillStyle = rgba(sig.color, sigGlow * fade);
            ctx.fill();
            // Core dot
            ctx.beginPath();
            ctx.arc(sx, sy, 2, 0, Math.PI * 2);
            ctx.fillStyle = rgba(sig.color, sigCore * fade);
            ctx.fill();
        }
    }

    // ---- Frequency Spectrum layer ----

    function createFrequencyState(env, opts) {
        opts = opts || {};
        var w = env.w, h = env.h;
        var barCount = opts.barCount || (isMobile ? 40 : 70);
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
        return { bars: bars, barWidth: Math.max(2, w / barCount * 0.4) };
    }

    function drawFrequency(ctx, w, h, fs, t, opts) {
        opts = opts || {};
        var glowOp = opts.glowOpacity !== undefined ? opts.glowOpacity : 0.1;
        var fillOp = opts.fillOpacity !== undefined ? opts.fillOpacity : 0.2;
        var borderOp = opts.borderOpacity !== undefined ? opts.borderOpacity : 0.6;
        var ts = opts.timeScale !== undefined ? opts.timeScale : 1.0;

        for (var i = 0; i < fs.bars.length; i++) {
            var bar = fs.bars[i];
            var oscillation = (Math.sin(t * ts * bar.freq + bar.phase) + 1) / 2;
            var waveEffect = (Math.sin(t * ts * 0.5 + i * 0.15) + 1) / 2;
            var barH = h * bar.maxH * (oscillation * 0.6 + waveEffect * 0.4);
            var barY = h - barH;
            var bw = fs.barWidth;

            // Glow
            ctx.fillStyle = rgba(bar.color, glowOp);
            ctx.beginPath();
            ctx.roundRect(bar.x - bw / 2 - 2, barY - 2, bw + 4, barH + 4, bw / 2 + 2);
            ctx.fill();

            // Fill
            ctx.fillStyle = rgba(bar.color, fillOp);
            ctx.beginPath();
            ctx.roundRect(bar.x - bw / 2, barY, bw, barH, bw / 2);
            ctx.fill();

            // Border
            ctx.strokeStyle = rgba(bar.color, borderOp);
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.roundRect(bar.x - bw / 2, barY, bw, barH, bw / 2);
            ctx.stroke();
        }
    }

    // ---- Stacked Curves layer ----

    function createStackedState(env) {
        var layers = [
            { name: 'nuclear', color: RES.nuclear, baseY: 0.88, amp: 0.02, freq: 0.5, phase: 0 },
            { name: 'hydro',   color: RES.hydro,   baseY: 0.78, amp: 0.04, freq: 0.8, phase: 1 },
            { name: 'solar',   color: RES.solar,   baseY: 0.55, amp: 0.12, freq: 1.2, phase: 0.5 },
            { name: 'wind',    color: RES.wind,    baseY: 0.42, amp: 0.08, freq: 2.0, phase: 2 },
            { name: 'storage', color: RES.storage, baseY: 0.35, amp: 0.05, freq: 1.5, phase: 3 }
        ];
        return { layers: layers };
    }

    function drawStackedCurves(ctx, w, h, ss, t, opts) {
        opts = opts || {};
        var gridOp = opts.gridOpacity !== undefined ? opts.gridOpacity : 0.03;
        var fillOp = opts.fillOpacity !== undefined ? opts.fillOpacity : 0.12;
        var strokeOp = opts.strokeOpacity !== undefined ? opts.strokeOpacity : 0.6;
        var glowOp = opts.glowOpacity !== undefined ? opts.glowOpacity : 0.15;
        var showParticles = opts.showParticles !== undefined ? opts.showParticles : true;
        var gridColor = opts.gridColor || 'rgba(255,255,255,';
        var particleColor = opts.particleColor || 'rgba(255,255,255,';

        // Subtle time-axis grid
        ctx.strokeStyle = gridColor + gridOp + ')';
        ctx.lineWidth = 0.5;
        for (var gx = 0; gx < w; gx += w / 24) {
            ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
        }

        // Draw stacked areas bottom-up
        for (var li = 0; li < ss.layers.length; li++) {
            var layer = ss.layers[li];
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
            ctx.fillStyle = rgba(layer.color, fillOp);
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
            ctx.strokeStyle = rgba(layer.color, strokeOp);
            ctx.lineWidth = 1.2;
            ctx.stroke();
            ctx.strokeStyle = rgba(layer.color, glowOp);
            ctx.lineWidth = 4;
            ctx.stroke();
        }

        // Occasional floating particles
        if (showParticles) {
            var particleCount = 12;
            for (var pi = 0; pi < particleCount; pi++) {
                var ppx = (pi / particleCount * w + t * 15) % w;
                var ppy = h * (0.3 + 0.5 * Math.sin(pi * 2.7 + t * 0.2));
                var pOp = 0.15 + 0.1 * Math.sin(t * 0.5 + pi);
                ctx.beginPath();
                ctx.arc(ppx, ppy - t * 3 % 20, 1.5, 0, Math.PI * 2);
                ctx.fillStyle = particleColor + pOp + ')';
                ctx.fill();
            }
        }
    }


    // ====================================================================
    // BANNER REGISTRATIONS
    // ====================================================================

    // ---- 1. Dense Circuit Map (light, warm bg #F5F5F3) ----
    registerBanner('dense-circuit-map', 'light', {
        create: function(env) {
            var cs = createCircuitState(env, {});
            return { ctx: env.ctx, w: env.w, h: env.h, circuit: cs };
        },
        draw: function(st, ts) {
            st.ctx.clearRect(0, 0, st.w, st.h);
            drawCircuit(st.ctx, st.w, st.h, st.circuit, ts / 1000, {});
        }
    });

    // ---- 2. Dense Circuit White (light, pure white bg) ----
    registerBanner('dense-circuit-white', 'light', {
        create: function(env) {
            var cs = createCircuitState(env, {});
            return { ctx: env.ctx, w: env.w, h: env.h, circuit: cs };
        },
        draw: function(st, ts) {
            st.ctx.clearRect(0, 0, st.w, st.h);
            drawCircuit(st.ctx, st.w, st.h, st.circuit, ts / 1000, {});
        }
    });

    // ---- 3. Dense Circuit Navy (dark, navy bg #0B1120) ----
    registerBanner('dense-circuit-navy', 'dark', {
        create: function(env) {
            var cs = createCircuitState(env, {});
            return { ctx: env.ctx, w: env.w, h: env.h, circuit: cs };
        },
        draw: function(st, ts) {
            st.ctx.clearRect(0, 0, st.w, st.h);
            drawCircuit(st.ctx, st.w, st.h, st.circuit, ts / 1000, {
                pathOpacity: 0.25, nodeOpacity: 0.55, ringOpacity: 0.6,
                signalGlowOpacity: 0.25, signalCoreOpacity: 0.9
            });
        }
    });

    // ---- 4. Frequency Spectrum (dark, navy bg #0B1120) ----
    registerBanner('frequency-spectrum', 'dark', {
        create: function(env) {
            var fs = createFrequencyState(env, {});
            return { ctx: env.ctx, w: env.w, h: env.h, freq: fs };
        },
        draw: function(st, ts) {
            st.ctx.clearRect(0, 0, st.w, st.h);
            drawFrequency(st.ctx, st.w, st.h, st.freq, ts / 1000, {});
        }
    });

    // ---- 5. Frequency Light (light, warm bg #F5F5F3) ----
    registerBanner('frequency-light', 'light', {
        create: function(env) {
            var fs = createFrequencyState(env, {});
            return { ctx: env.ctx, w: env.w, h: env.h, freq: fs };
        },
        draw: function(st, ts) {
            st.ctx.clearRect(0, 0, st.w, st.h);
            drawFrequency(st.ctx, st.w, st.h, st.freq, ts / 1000, {
                glowOpacity: 0.08, fillOpacity: 0.25, borderOpacity: 0.5
            });
        }
    });

    // ---- 6. Stacked Curves Dark (dark, navy bg #0B1120) ----
    registerBanner('stacked-curves-dark', 'dark', {
        create: function(env) {
            var ss = createStackedState(env);
            return { ctx: env.ctx, w: env.w, h: env.h, stacked: ss };
        },
        draw: function(st, ts) {
            st.ctx.clearRect(0, 0, st.w, st.h);
            drawStackedCurves(st.ctx, st.w, st.h, st.stacked, ts / 1000, {});
        }
    });

    // ---- 7. Network Constellation Light (light) ----
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
                var colors = [RES.nuclear, RES.hydro, RES.solar, RES.wind, RES.battery, RES.ldes];
                var col = colors[Math.floor(rng() * colors.length)];
                nodes.push({ x: nx, y: ny, r: 2 + rng() * 4, color: col, baseX: nx, baseY: ny });
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


    // ====================================================================
    // COMPOSITE BANNERS (layer multiple animations)
    // ====================================================================

    // ---- 8. Frequency + Dense Circuit (dark, navy bg) ----
    registerBanner('frequency-dense-navy', 'dark', {
        create: function(env) {
            var cs = createCircuitState(env, {});
            var fs = createFrequencyState(env, {});
            return { ctx: env.ctx, w: env.w, h: env.h, circuit: cs, freq: fs };
        },
        draw: function(st, ts) {
            var t = ts / 1000;
            st.ctx.clearRect(0, 0, st.w, st.h);
            // Base layer: circuit (subdued)
            drawCircuit(st.ctx, st.w, st.h, st.circuit, t, {
                pathOpacity: 0.15, nodeOpacity: 0.3, ringOpacity: 0.35,
                signalGlowOpacity: 0.12, signalCoreOpacity: 0.5
            });
            // Top layer: frequency bars (slowed to stacked-curve pace)
            drawFrequency(st.ctx, st.w, st.h, st.freq, t, {
                timeScale: 0.35
            });
        }
    });

    // ---- 9. Frequency + Dense Circuit (light, white bg) ----
    registerBanner('frequency-dense-white', 'light', {
        create: function(env) {
            var cs = createCircuitState(env, {});
            var fs = createFrequencyState(env, {});
            return { ctx: env.ctx, w: env.w, h: env.h, circuit: cs, freq: fs };
        },
        draw: function(st, ts) {
            var t = ts / 1000;
            st.ctx.clearRect(0, 0, st.w, st.h);
            // Base layer: circuit (subdued for light bg)
            drawCircuit(st.ctx, st.w, st.h, st.circuit, t, {
                pathOpacity: 0.08, nodeOpacity: 0.2, ringOpacity: 0.25,
                signalGlowOpacity: 0.08, signalCoreOpacity: 0.4
            });
            // Top layer: frequency bars (light-adjusted, slowed to stacked-curve pace)
            drawFrequency(st.ctx, st.w, st.h, st.freq, t, {
                glowOpacity: 0.08, fillOpacity: 0.25, borderOpacity: 0.5,
                timeScale: 0.35
            });
        }
    });

    // ---- 10. Stacked Curves + Dense Circuit (dark, navy bg) ----
    registerBanner('stacked-dense-navy', 'dark', {
        create: function(env) {
            var cs = createCircuitState(env, {});
            var ss = createStackedState(env);
            return { ctx: env.ctx, w: env.w, h: env.h, circuit: cs, stacked: ss };
        },
        draw: function(st, ts) {
            var t = ts / 1000;
            st.ctx.clearRect(0, 0, st.w, st.h);
            // Base layer: circuit (very subdued)
            drawCircuit(st.ctx, st.w, st.h, st.circuit, t, {
                pathOpacity: 0.12, nodeOpacity: 0.25, ringOpacity: 0.3,
                signalGlowOpacity: 0.1, signalCoreOpacity: 0.4
            });
            // Top layer: stacked curves
            drawStackedCurves(st.ctx, st.w, st.h, st.stacked, t, {});
        }
    });

    // ---- 11. Stacked Curves + Dense Circuit (light, white bg) ----
    registerBanner('stacked-dense-white', 'light', {
        create: function(env) {
            var cs = createCircuitState(env, {});
            var ss = createStackedState(env);
            return { ctx: env.ctx, w: env.w, h: env.h, circuit: cs, stacked: ss };
        },
        draw: function(st, ts) {
            var t = ts / 1000;
            st.ctx.clearRect(0, 0, st.w, st.h);
            // Base layer: circuit (subdued for light bg)
            drawCircuit(st.ctx, st.w, st.h, st.circuit, t, {
                pathOpacity: 0.08, nodeOpacity: 0.2, ringOpacity: 0.25,
                signalGlowOpacity: 0.08, signalCoreOpacity: 0.4
            });
            // Top layer: stacked curves (light-adjusted)
            drawStackedCurves(st.ctx, st.w, st.h, st.stacked, t, {
                gridOpacity: 0.04, fillOpacity: 0.18, strokeOpacity: 0.5,
                glowOpacity: 0.12, showParticles: false,
                gridColor: 'rgba(30,41,59,'
            });
        }
    });

})();
