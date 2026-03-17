// ============================================================================
// CANVAS BANNERS - Light Frequency + Circuit hero banner variants
// ============================================================================
// Loaded on pages that need canvas-based banners (e.g., about.html).
// Registers variants into window._canvasBanners for shared-header.js to use.
//
// Variants (all light freq+circuit, different background tones):
//   freq-circuit-flat      - flat #FEFEFE (near-white, neutral)
//   freq-circuit-warm      - flat #FEFDFC (warm off-white)
//   freq-circuit-cool      - flat #FEFEFD (cool off-white)
//   freq-circuit-gradient  - gradient blend of all three
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

                // IntersectionObserver - pause when offscreen
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


    // ====================================================================
    // REUSABLE LAYER FUNCTIONS
    // ====================================================================

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
            } else {
                ctx.beginPath();
                ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
                ctx.strokeStyle = rgba(n.color, ringOp);
                ctx.lineWidth = 1;
                ctx.stroke();
            }
        }

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
            ctx.beginPath();
            ctx.arc(sx, sy, 4, 0, Math.PI * 2);
            ctx.fillStyle = rgba(sig.color, sigGlow * fade);
            ctx.fill();
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

            ctx.fillStyle = rgba(bar.color, glowOp);
            ctx.beginPath();
            ctx.roundRect(bar.x - bw / 2 - 2, barY - 2, bw + 4, barH + 4, bw / 2 + 2);
            ctx.fill();

            ctx.fillStyle = rgba(bar.color, fillOp);
            ctx.beginPath();
            ctx.roundRect(bar.x - bw / 2, barY, bw, barH, bw / 2);
            ctx.fill();

            ctx.strokeStyle = rgba(bar.color, borderOp);
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.roundRect(bar.x - bw / 2, barY, bw, barH, bw / 2);
            ctx.stroke();
        }
    }


    // ====================================================================
    // BANNER REGISTRATIONS - Light Freq + Circuit with background variants
    // ====================================================================

    // Shared draw function for all 4 variants (same animation, different bg)
    function drawFreqCircuitLight(st, ts) {
        var t = ts / 1000;
        st.ctx.clearRect(0, 0, st.w, st.h);
        // Base layer: circuit (subdued for light bg)
        drawCircuit(st.ctx, st.w, st.h, st.circuit, t, {
            pathOpacity: 0.08, nodeOpacity: 0.2, ringOpacity: 0.25,
            signalGlowOpacity: 0.08, signalCoreOpacity: 0.4
        });
        // Top layer: frequency bars (light-adjusted, slowed)
        drawFrequency(st.ctx, st.w, st.h, st.freq, t, {
            glowOpacity: 0.08, fillOpacity: 0.25, borderOpacity: 0.5,
            timeScale: 0.35
        });
    }

    function createFreqCircuitState(env) {
        var cs = createCircuitState(env, {});
        var fs = createFrequencyState(env, {});
        return { ctx: env.ctx, w: env.w, h: env.h, circuit: cs, freq: fs };
    }

    // ---- 1. Flat #FEFEFE (near-white, neutral) ----
    registerBanner('freq-circuit-flat', 'light', {
        create: createFreqCircuitState,
        draw: drawFreqCircuitLight
    });

    // ---- 2. Flat #FEFDFC (warm off-white) ----
    registerBanner('freq-circuit-warm', 'light', {
        create: createFreqCircuitState,
        draw: drawFreqCircuitLight
    });

    // ---- 3. Flat #FEFEFD (cool off-white) ----
    registerBanner('freq-circuit-cool', 'light', {
        create: createFreqCircuitState,
        draw: drawFreqCircuitLight
    });

    // ---- 4. Gradient blend (#FEFDFC → #FEFEFE → #FEFEFD) ----
    registerBanner('freq-circuit-gradient', 'light', {
        create: createFreqCircuitState,
        draw: drawFreqCircuitLight
    });

})();
