// ============================================================================
// SHARED HEADER — Injects the page banner with SVG waveform/heartbeat overlay
// ============================================================================
// Usage: Include this script AFTER shared.css and nav.js
//   <script src="js/shared-header.js"></script>
//
// Then place a simple header element in your HTML:
//   <header class="header" id="pageHeader" data-header-variant="default">
//       <h1>Page Title</h1>
//       <div class="subtitle">Page description</div>
//       <div class="header-accent"></div>
//   </header>
//
// SVG variants: default (Midnight Grid)
// Canvas variants: frosted-circuit, particle-fill-dark, particle-fill-light,
//   dense-circuit-map, heatmap-pulse-dark, heatmap-pulse-light, stacked-curves-dark,
//   stacked-curves-light, network-constellation-dark, network-constellation-light,
//   frequency-spectrum, topographic-contours-light, topographic-contours-dark
// Set via data-header-variant attribute. CSS class header--{variant} is auto-added.
// ============================================================================

(function() {
    'use strict';

    // ---- Shared SVG building blocks ----

    var GRID_DOTS_DARK = [
        '<circle cx="200" cy="80" r="1.5" fill="rgba(255,255,255,0.12)"/>',
        '<circle cx="400" cy="60" r="1.8" fill="rgba(255,255,255,0.10)"/>',
        '<circle cx="600" cy="100" r="1.5" fill="rgba(255,255,255,0.11)"/>',
        '<circle cx="800" cy="50" r="2.0" fill="rgba(255,255,255,0.09)"/>',
        '<circle cx="1000" cy="75" r="1.5" fill="rgba(255,255,255,0.12)"/>',
        '<circle cx="1200" cy="90" r="1.8" fill="rgba(255,255,255,0.10)"/>',
        '<circle cx="300" cy="250" r="1.5" fill="rgba(255,255,255,0.08)"/>',
        '<circle cx="700" cy="240" r="1.8" fill="rgba(255,255,255,0.10)"/>',
        '<circle cx="1100" cy="260" r="1.5" fill="rgba(255,255,255,0.11)"/>'
    ].join('\n');

    var GRID_LINES_DARK = [
        '<line x1="0" y1="70" x2="1440" y2="70" stroke="rgba(255,255,255,0.04)" stroke-width="0.5"/>',
        '<line x1="0" y1="140" x2="1440" y2="140" stroke="rgba(255,255,255,0.05)" stroke-width="0.5"/>',
        '<line x1="0" y1="210" x2="1440" y2="210" stroke="rgba(255,255,255,0.04)" stroke-width="0.5"/>'
    ].join('\n');

    // Standard heartbeat paths (reused across variants)
    var HEARTBEAT_RED = [
        '<path d="M0,140 L180,140 L200,140 L210,138 L218,142 ',
        'L225,120 L232,165 L240,95 L248,170 L255,115 L262,145 L270,140 ',
        'L350,140 L450,140 L460,138 L468,142 ',
        'L475,118 L482,162 L490,90 L498,168 L505,112 L512,148 L520,140 ',
        'L650,140 L750,140 L760,138 L768,142 ',
        'L775,122 L782,160 L790,98 L798,165 L805,118 L812,146 L820,140 ',
        'L950,140 L1050,140 L1060,138 L1068,142 ',
        'L1075,120 L1082,164 L1090,92 L1098,170 L1105,114 L1112,148 L1120,140 ',
        'L1250,140 L1350,140 L1360,138 L1368,142 ',
        'L1375,124 L1382,158 L1390,100 L1398,166 L1405,120 L1412,144 L1420,140 ',
        'L1440,140"'
    ].join('');

    var HEARTBEAT_CYAN = [
        '<path d="M-100,155 L80,155 L100,155 L108,153 L115,157 ',
        'L122,135 L129,178 L137,105 L145,180 L152,130 L159,158 L167,155 ',
        'L300,155 L400,155 L408,153 L415,157 ',
        'L422,133 L429,175 L437,100 L445,178 L452,128 L459,160 L467,155 ',
        'L600,155 L700,155 L708,153 L715,157 ',
        'L722,138 L729,172 L737,108 L745,176 L752,132 L759,158 L767,155 ',
        'L900,155 L1000,155 L1008,153 L1015,157 ',
        'L1022,136 L1029,176 L1037,103 L1045,179 L1052,130 L1059,160 L1067,155 ',
        'L1200,155 L1300,155 L1308,153 L1315,157 ',
        'L1322,140 L1329,170 L1337,110 L1345,174 L1352,134 L1359,156 L1367,155 ',
        'L1440,155"'
    ].join('');

    // Shared wave path data
    var WAVE_HYDRO = {
        fill: 'M0,200 C120,180 240,160 360,170 C480,180 600,200 720,190 C840,180 960,160 1080,170 C1200,180 1320,200 1440,190 L1440,280 L0,280 Z',
        fillAlt: 'M0,195 C120,175 240,165 360,175 C480,185 600,195 720,185 C840,175 960,165 1080,175 C1200,185 1320,195 1440,185 L1440,280 L0,280 Z',
        stroke: 'M0,200 C120,180 240,160 360,170 C480,180 600,200 720,190 C840,180 960,160 1080,170 C1200,180 1320,200 1440,190',
        strokeAlt: 'M0,195 C120,175 240,165 360,175 C480,185 600,195 720,185 C840,175 960,165 1080,175 C1200,185 1320,195 1440,185'
    };
    var WAVE_SOLAR = {
        fill: 'M0,240 C180,230 300,180 450,140 C600,100 720,90 900,130 C1050,165 1200,210 1350,230 L1440,240 L1440,280 L0,280 Z',
        fillAlt: 'M0,235 C180,225 300,175 450,145 C600,105 720,95 900,125 C1050,160 1200,205 1350,225 L1440,235 L1440,280 L0,280 Z',
        stroke: 'M0,240 C180,230 300,180 450,140 C600,100 720,90 900,130 C1050,165 1200,210 1350,230 L1440,240',
        strokeAlt: 'M0,235 C180,225 300,175 450,145 C600,105 720,95 900,125 C1050,160 1200,205 1350,225 L1440,235'
    };
    var WAVE_WIND = {
        fill: 'M0,180 C80,165 160,190 280,160 C400,130 480,170 600,150 C720,130 840,160 960,140 C1080,120 1200,155 1320,145 L1440,160 L1440,280 L0,280 Z',
        fillAlt: 'M0,175 C80,160 160,185 280,155 C400,135 480,165 600,155 C720,135 840,155 960,135 C1080,125 1200,150 1320,140 L1440,155 L1440,280 L0,280 Z',
        stroke: 'M0,180 C80,165 160,190 280,160 C400,130 480,170 600,150 C720,130 840,160 960,140 C1080,120 1200,155 1320,145 L1440,160',
        strokeAlt: 'M0,175 C80,160 160,185 280,155 C400,135 480,165 600,155 C720,135 840,155 960,135 C1080,125 1200,150 1320,140 L1440,155'
    };
    var WAVE_DEMAND = {
        stroke: 'M0,220 C240,215 480,225 720,218 C960,211 1200,222 1440,216',
        strokeAlt: 'M0,222 C240,217 480,222 720,215 C960,213 1200,220 1440,218'
    };

    // ---- Helper to build animated wave paths ----
    function animWave(wave, fillGrad, fillOpacity, strokeColor, strokeWidth, dur) {
        var parts = [];
        if (fillGrad) {
            parts.push(
                '<path d="' + wave.fill + '" fill="url(#' + fillGrad + ')" opacity="' + fillOpacity + '">',
                '  <animate attributeName="d" dur="' + dur + 's" repeatCount="indefinite" values="' +
                    wave.fill + ';' + wave.fillAlt + ';' + wave.fill + '"/>',
                '</path>'
            );
        }
        var sPath = wave.stroke || wave.fill.replace(/ L1440,280 L0,280 Z/, '');
        var sAlt = wave.strokeAlt || wave.fillAlt.replace(/ L1440,280 L0,280 Z/, '');
        parts.push(
            '<path d="' + sPath + '" fill="none" stroke="' + strokeColor + '" stroke-width="' + strokeWidth + '">',
            '  <animate attributeName="d" dur="' + dur + 's" repeatCount="indefinite" values="' +
                sPath + ';' + sAlt + ';' + sPath + '"/>',
            '</path>'
        );
        return parts.join('\n');
    }

    function heartbeat(path, strokeColor, strokeWidth, dur, opLow, opHigh) {
        return [
            path,
            ' fill="none" stroke="' + strokeColor + '" stroke-width="' + strokeWidth + '"',
            ' stroke-linecap="round" stroke-linejoin="round">',
            '  <animate attributeName="stroke-opacity" dur="' + dur + 's" repeatCount="indefinite"',
            '    values="' + opLow + ';' + opHigh + ';' + opLow + '" keyTimes="0;0.5;1"/>',
            '</path>'
        ].join('');
    }

    // ---- SVG variant generators ----

    var VARIANTS = {};

    // ========== 1. DEFAULT (Midnight Grid) ==========
    VARIANTS['default'] = function() {
        return [
            '<defs>',
            '  <linearGradient id="hdr-blue-fade" x1="0" y1="0" x2="0" y2="1">',
            '    <stop offset="0%" stop-color="rgba(14,165,233,0.18)"/>',
            '    <stop offset="100%" stop-color="rgba(14,165,233,0)"/>',
            '  </linearGradient>',
            '  <linearGradient id="hdr-amber-fade" x1="0" y1="0" x2="0" y2="1">',
            '    <stop offset="0%" stop-color="rgba(245,158,11,0.14)"/>',
            '    <stop offset="100%" stop-color="rgba(245,158,11,0)"/>',
            '  </linearGradient>',
            '  <linearGradient id="hdr-green-fade" x1="0" y1="0" x2="0" y2="1">',
            '    <stop offset="0%" stop-color="rgba(34,197,94,0.12)"/>',
            '    <stop offset="100%" stop-color="rgba(34,197,94,0)"/>',
            '  </linearGradient>',
            '</defs>',
            animWave(WAVE_HYDRO, 'hdr-blue-fade', '0.35', 'rgba(14,165,233,0.35)', '1.8', 12),
            animWave(WAVE_SOLAR, 'hdr-amber-fade', '0.30', 'rgba(245,158,11,0.30)', '1.5', 15),
            animWave(WAVE_WIND, 'hdr-green-fade', '0.25', 'rgba(34,197,94,0.25)', '1.2', 18),
            // Demand baseline
            '<path d="' + WAVE_DEMAND.stroke + '" fill="none" stroke="rgba(255,255,255,0.12)" stroke-width="2.0">',
            '  <animate attributeName="d" dur="20s" repeatCount="indefinite" values="' +
                WAVE_DEMAND.stroke + ';' + WAVE_DEMAND.strokeAlt + ';' + WAVE_DEMAND.stroke + '"/>',
            '</path>',
            heartbeat(HEARTBEAT_RED, 'rgba(239,68,68,0.18)', '1.5', 3, '0.18', '0.30'),
            heartbeat(HEARTBEAT_CYAN, 'rgba(56,189,248,0.14)', '1.2', 4, '0.14', '0.24'),
            GRID_DOTS_DARK,
            GRID_LINES_DARK
        ].join('\n');
    };

    // (Old SVG variants frosted/terrain/hexmosaic removed — replaced by canvas banners in canvas-banners.js)

    // ---- Build SVG wrapper ----
    function buildSVG(variant) {
        var gen = VARIANTS[variant] || VARIANTS['default'];
        return [
            '<div class="header-svg-overlay">',
            '<svg viewBox="0 0 1440 280" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">',
            gen(),
            '</svg>',
            '</div>'
        ].join('\n');
    }

    // ---- Active canvas banner tracking ----
    var _activeCanvasBanner = null;

    // ---- Helper: check if variant is a canvas banner ----
    function isCanvasVariant(variant) {
        return window._canvasBanners && window._canvasBanners[variant];
    }

    // ---- Helper: get theme for a canvas variant ----
    function getCanvasTheme(variant) {
        if (isCanvasVariant(variant)) {
            return window._canvasBanners[variant].theme || 'dark';
        }
        return 'dark';
    }

    // ---- Helper: init a canvas banner inside a header ----
    function initCanvasBanner(header, variant) {
        var wrapper = document.createElement('div');
        wrapper.className = 'header-canvas-overlay';
        var canvas = document.createElement('canvas');
        wrapper.appendChild(canvas);
        header.insertBefore(wrapper, header.firstChild);

        // Size canvas with DPR
        var dpr = window.devicePixelRatio || 1;
        var rect = header.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';

        var bannerDef = window._canvasBanners[variant];
        bannerDef.init(canvas, header);
        _activeCanvasBanner = bannerDef;
    }

    // ---- Cleanup any active canvas banner ----
    function destroyCanvasBanner() {
        if (_activeCanvasBanner) {
            _activeCanvasBanner.destroy();
            _activeCanvasBanner = null;
        }
    }

    // ---- Inject into headers ----
    function injectOverlay() {
        var headers = document.querySelectorAll('.header');
        headers.forEach(function(header) {
            if (header.querySelector('.header-svg-overlay') || header.querySelector('.header-canvas-overlay')) return;
            var variant = header.getAttribute('data-header-variant') || 'default';

            if (isCanvasVariant(variant)) {
                // Canvas banner
                var theme = getCanvasTheme(variant);
                header.classList.add('header--' + theme);
                initCanvasBanner(header, variant);
            } else {
                // SVG banner (default)
                if (variant !== 'default') {
                    header.classList.add('header--' + variant);
                }
                header.insertAdjacentHTML('afterbegin', buildSVG(variant));
            }
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', injectOverlay);
    } else {
        injectOverlay();
    }

    // Expose variant list (dynamic — includes canvas banners when loaded)
    Object.defineProperty(window, '_headerVariants', {
        get: function() {
            var keys = Object.keys(VARIANTS);
            if (window._canvasBanners) {
                keys = keys.concat(Object.keys(window._canvasBanners));
            }
            return keys;
        }
    });

    // Expose for dynamic switching (used by variant selector on test pages)
    window._switchHeaderVariant = function(variant) {
        var headers = document.querySelectorAll('.header');
        headers.forEach(function(header) {
            // 1. Destroy active canvas banner if any
            destroyCanvasBanner();

            // 2. Remove old overlays
            var oldSvg = header.querySelector('.header-svg-overlay');
            if (oldSvg) oldSvg.remove();
            var oldCanvas = header.querySelector('.header-canvas-overlay');
            if (oldCanvas) oldCanvas.remove();

            // 3. Remove all theme/variant classes
            header.classList.remove('header--light', 'header--dark');
            // Also remove any legacy classes
            Object.keys(VARIANTS).forEach(function(v) {
                header.classList.remove('header--' + v);
            });

            // 4. Set variant attribute
            header.setAttribute('data-header-variant', variant);

            // 5. Apply new variant
            if (isCanvasVariant(variant)) {
                var theme = getCanvasTheme(variant);
                header.classList.add('header--' + theme);
                initCanvasBanner(header, variant);
            } else {
                // SVG (default)
                header.insertAdjacentHTML('afterbegin', buildSVG(variant));
            }
        });
    };
})();
