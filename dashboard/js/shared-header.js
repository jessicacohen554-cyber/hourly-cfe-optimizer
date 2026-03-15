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
// Variants: default | frosted | terrain | hexmosaic
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

    // ========== 4. FROSTED CIRCUIT (Light/Glass) ==========
    VARIANTS['frosted'] = function() {
        // Higher-opacity curves for light background, grid dots in dark
        var gridDotsLight = [
            '<circle cx="200" cy="80" r="1.5" fill="rgba(30,55,100,0.10)"/>',
            '<circle cx="400" cy="60" r="1.8" fill="rgba(30,55,100,0.08)"/>',
            '<circle cx="600" cy="100" r="1.5" fill="rgba(30,55,100,0.09)"/>',
            '<circle cx="800" cy="50" r="2.0" fill="rgba(30,55,100,0.07)"/>',
            '<circle cx="1000" cy="75" r="1.5" fill="rgba(30,55,100,0.10)"/>',
            '<circle cx="1200" cy="90" r="1.8" fill="rgba(30,55,100,0.08)"/>',
            '<circle cx="300" cy="250" r="1.5" fill="rgba(30,55,100,0.06)"/>',
            '<circle cx="700" cy="240" r="1.8" fill="rgba(30,55,100,0.08)"/>',
            '<circle cx="1100" cy="260" r="1.5" fill="rgba(30,55,100,0.09)"/>'
        ].join('\n');

        var gridLinesLight = [
            '<line x1="0" y1="70" x2="1440" y2="70" stroke="rgba(148,163,184,0.12)" stroke-width="0.5"/>',
            '<line x1="0" y1="140" x2="1440" y2="140" stroke="rgba(148,163,184,0.15)" stroke-width="0.5"/>',
            '<line x1="0" y1="210" x2="1440" y2="210" stroke="rgba(148,163,184,0.12)" stroke-width="0.5"/>'
        ].join('\n');

        // Dispatch event dots where curves cross grid lines
        var dispatchDots = [
            '<circle cx="360" cy="170" r="3" fill="rgba(14,165,233,0.25)" stroke="rgba(14,165,233,0.40)" stroke-width="1"/>',
            '<circle cx="720" cy="90" r="3" fill="rgba(245,158,11,0.25)" stroke="rgba(245,158,11,0.40)" stroke-width="1"/>',
            '<circle cx="960" cy="140" r="3" fill="rgba(34,197,94,0.25)" stroke="rgba(34,197,94,0.40)" stroke-width="1"/>',
            '<circle cx="480" cy="170" r="2.5" fill="rgba(245,158,11,0.20)" stroke="rgba(245,158,11,0.35)" stroke-width="1"/>',
            '<circle cx="1200" cy="155" r="2.5" fill="rgba(14,165,233,0.20)" stroke="rgba(14,165,233,0.35)" stroke-width="1"/>',
            '<circle cx="280" cy="160" r="2.5" fill="rgba(34,197,94,0.20)" stroke="rgba(34,197,94,0.35)" stroke-width="1"/>'
        ].join('\n');

        return [
            '<defs>',
            '  <linearGradient id="hdr-fr-blue" x1="0" y1="0" x2="0" y2="1">',
            '    <stop offset="0%" stop-color="rgba(14,165,233,0.25)"/>',
            '    <stop offset="100%" stop-color="rgba(14,165,233,0.02)"/>',
            '  </linearGradient>',
            '  <linearGradient id="hdr-fr-amber" x1="0" y1="0" x2="0" y2="1">',
            '    <stop offset="0%" stop-color="rgba(245,158,11,0.20)"/>',
            '    <stop offset="100%" stop-color="rgba(245,158,11,0.02)"/>',
            '  </linearGradient>',
            '  <linearGradient id="hdr-fr-green" x1="0" y1="0" x2="0" y2="1">',
            '    <stop offset="0%" stop-color="rgba(34,197,94,0.18)"/>',
            '    <stop offset="100%" stop-color="rgba(34,197,94,0.02)"/>',
            '  </linearGradient>',
            '</defs>',
            // Full-color energy curves — high opacity for light bg
            animWave(WAVE_HYDRO, 'hdr-fr-blue', '0.50', 'rgba(14,165,233,0.55)', '2.0', 12),
            animWave(WAVE_SOLAR, 'hdr-fr-amber', '0.40', 'rgba(245,158,11,0.50)', '1.8', 15),
            animWave(WAVE_WIND, 'hdr-fr-green', '0.35', 'rgba(34,197,94,0.45)', '1.5', 18),
            // Demand — dark gray
            '<path d="' + WAVE_DEMAND.stroke + '" fill="none" stroke="rgba(26,39,68,0.15)" stroke-width="1.8">',
            '  <animate attributeName="d" dur="20s" repeatCount="indefinite" values="' +
                WAVE_DEMAND.stroke + ';' + WAVE_DEMAND.strokeAlt + ';' + WAVE_DEMAND.stroke + '"/>',
            '</path>',
            // Navy heartbeat
            heartbeat(HEARTBEAT_RED, 'rgba(26,39,68,0.20)', '1.5', 3, '0.20', '0.40'),
            dispatchDots,
            gridDotsLight,
            gridLinesLight
        ].join('\n');
    };

    // ========== 12. STACKED TERRAIN — LIGHT (Dispatch Art) ==========
    VARIANTS['terrain'] = function() {
        var hydro = {
            fill: 'M0,245 C120,238 240,232 360,236 C480,240 600,248 720,242 C840,236 960,230 1080,235 C1200,240 1320,246 1440,242 L1440,280 L0,280 Z',
            fillAlt: 'M0,242 C120,235 240,235 360,238 C480,241 600,245 720,240 C840,234 960,233 1080,237 C1200,241 1320,244 1440,240 L1440,280 L0,280 Z',
            stroke: 'M0,245 C120,238 240,232 360,236 C480,240 600,248 720,242 C840,236 960,230 1080,235 C1200,240 1320,246 1440,242',
            strokeAlt: 'M0,242 C120,235 240,235 360,238 C480,241 600,245 720,240 C840,234 960,233 1080,237 C1200,241 1320,244 1440,240'
        };
        var wind = {
            fill: 'M0,215 C80,205 160,220 280,200 C400,185 520,210 640,195 C760,180 880,200 1000,188 C1120,175 1240,195 1360,190 L1440,195 L1440,280 L0,280 Z',
            fillAlt: 'M0,212 C80,200 160,215 280,198 C400,188 520,205 640,192 C760,178 880,196 1000,185 C1120,178 1240,192 1360,188 L1440,192 L1440,280 L0,280 Z',
            stroke: 'M0,215 C80,205 160,220 280,200 C400,185 520,210 640,195 C760,180 880,200 1000,188 C1120,175 1240,195 1360,190 L1440,195',
            strokeAlt: 'M0,212 C80,200 160,215 280,198 C400,188 520,205 640,192 C760,178 880,196 1000,185 C1120,178 1240,192 1360,188 L1440,192'
        };
        var solar = {
            fill: 'M0,210 C180,200 300,170 450,145 C600,125 720,120 900,140 C1050,158 1200,185 1350,200 L1440,205 L1440,280 L0,280 Z',
            fillAlt: 'M0,208 C180,198 300,168 450,148 C600,128 720,122 900,138 C1050,155 1200,182 1350,198 L1440,203 L1440,280 L0,280 Z',
            stroke: 'M0,210 C180,200 300,170 450,145 C600,125 720,120 900,140 C1050,158 1200,185 1350,200 L1440,205',
            strokeAlt: 'M0,208 C180,198 300,168 450,148 C600,128 720,122 900,138 C1050,155 1200,182 1350,198 L1440,203'
        };
        var storage = {
            fill: 'M0,205 C180,198 350,192 500,185 C650,178 780,175 920,178 C1060,182 1200,190 1350,196 L1440,200 L1440,280 L0,280 Z',
            fillAlt: 'M0,203 C180,196 350,190 500,183 C650,176 780,173 920,176 C1060,180 1200,188 1350,194 L1440,198 L1440,280 L0,280 Z',
            stroke: 'M0,205 C180,198 350,192 500,185 C650,178 780,175 920,178 C1060,182 1200,190 1350,196 L1440,200',
            strokeAlt: 'M0,203 C180,196 350,190 500,183 C650,176 780,173 920,176 C1060,180 1200,188 1350,194 L1440,198'
        };
        var demand = {
            stroke: 'M0,155 C240,148 480,160 720,152 C960,145 1200,155 1440,150',
            strokeAlt: 'M0,153 C240,146 480,158 720,150 C960,143 1200,153 1440,148'
        };
        return [
            '<defs>',
            '  <linearGradient id="hdr-ter-hydro" x1="0" y1="0" x2="0" y2="1">',
            '    <stop offset="0%" stop-color="rgba(14,165,233,0.30)"/>',
            '    <stop offset="100%" stop-color="rgba(14,165,233,0.05)"/>',
            '  </linearGradient>',
            '  <linearGradient id="hdr-ter-wind" x1="0" y1="0" x2="0" y2="1">',
            '    <stop offset="0%" stop-color="rgba(34,197,94,0.25)"/>',
            '    <stop offset="100%" stop-color="rgba(34,197,94,0.05)"/>',
            '  </linearGradient>',
            '  <linearGradient id="hdr-ter-solar" x1="0" y1="0" x2="0" y2="1">',
            '    <stop offset="0%" stop-color="rgba(245,158,11,0.30)"/>',
            '    <stop offset="100%" stop-color="rgba(245,158,11,0.05)"/>',
            '  </linearGradient>',
            '  <linearGradient id="hdr-ter-storage" x1="0" y1="0" x2="0" y2="1">',
            '    <stop offset="0%" stop-color="rgba(239,68,68,0.20)"/>',
            '    <stop offset="100%" stop-color="rgba(239,68,68,0.03)"/>',
            '  </linearGradient>',
            '</defs>',
            animWave(hydro, 'hdr-ter-hydro', '0.40', 'rgba(14,165,233,0.50)', '1.5', 18),
            animWave(wind, 'hdr-ter-wind', '0.35', 'rgba(34,197,94,0.45)', '1.2', 15),
            animWave(solar, 'hdr-ter-solar', '0.35', 'rgba(245,158,11,0.50)', '1.2', 12),
            animWave(storage, 'hdr-ter-storage', '0.30', 'rgba(239,68,68,0.40)', '1.0', 16),
            // Demand line — dark slate, prominent on light bg
            '<path d="' + demand.stroke + '" fill="none" stroke="rgba(30,41,59,0.35)" stroke-width="2" stroke-dasharray="6 3">',
            '  <animate attributeName="d" dur="20s" repeatCount="indefinite" values="' +
                demand.stroke + ';' + demand.strokeAlt + ';' + demand.stroke + '"/>',
            '</path>',
            '<text x="720" y="138" font-family="monospace" font-size="8" fill="rgba(71,85,105,0.30)" text-anchor="middle">the gap</text>'
        ].join('\n');
    };

    // ========== 13. SAND TERRAIN — LIGHT (Pebble-fill stacked dispatch) ==========
    VARIANTS['hexmosaic'] = function() {
        // Terrain curve evaluation — cubic Bezier segments from terrain variant paths
        var curves = {
            storage: [ // top layer (lowest y values)
                {x0:0,y0:205, cx1:180,cy1:198, cx2:350,cy2:192, x3:500,y3:185},
                {x0:500,y0:185, cx1:650,cy1:178, cx2:780,cy2:175, x3:920,y3:178},
                {x0:920,y0:178, cx1:1060,cy1:182, cx2:1200,cy2:190, x3:1350,y3:196},
                {x0:1350,y0:196, cx1:1395,cy1:198, cx2:1440,cy2:200, x3:1440,y3:200}
            ],
            solar: [
                {x0:0,y0:210, cx1:180,cy1:200, cx2:300,cy2:170, x3:450,y3:145},
                {x0:450,y0:145, cx1:600,cy1:125, cx2:720,cy2:120, x3:900,y3:140},
                {x0:900,y0:140, cx1:1050,cy1:158, cx2:1200,cy2:185, x3:1350,y3:200},
                {x0:1350,y0:200, cx1:1395,cy1:203, cx2:1440,cy2:205, x3:1440,y3:205}
            ],
            wind: [
                {x0:0,y0:215, cx1:80,cy1:205, cx2:160,cy2:220, x3:280,y3:200},
                {x0:280,y0:200, cx1:400,cy1:185, cx2:520,cy2:210, x3:640,y3:195},
                {x0:640,y0:195, cx1:760,cy1:180, cx2:880,cy2:200, x3:1000,y3:188},
                {x0:1000,y0:188, cx1:1120,cy1:175, cx2:1240,cy2:195, x3:1360,y3:190},
                {x0:1360,y0:190, cx1:1400,cy1:193, cx2:1440,cy2:195, x3:1440,y3:195}
            ],
            hydro: [
                {x0:0,y0:245, cx1:120,cy1:238, cx2:240,cy2:232, x3:360,y3:236},
                {x0:360,y0:236, cx1:480,cy1:240, cx2:600,cy2:248, x3:720,y3:242},
                {x0:720,y0:242, cx1:840,cy1:236, cx2:960,cy2:230, x3:1080,y3:235},
                {x0:1080,y0:235, cx1:1200,cy1:240, cx2:1320,cy2:246, x3:1440,y3:242}
            ]
        };

        function evalCurve(segs, x) {
            for (var i = 0; i < segs.length; i++) {
                var s = segs[i];
                if (x >= s.x0 && x <= s.x3) {
                    var t = (s.x3 === s.x0) ? 0 : (x - s.x0) / (s.x3 - s.x0);
                    var mt = 1 - t;
                    return mt*mt*mt*s.y0 + 3*mt*mt*t*s.cy1 + 3*mt*t*t*s.cy2 + t*t*t*s.y3;
                }
            }
            return segs[segs.length - 1].y3;
        }

        // Layers: bottom-to-top painting order (back to front)
        var layers = [
            { curve: 'hydro', color: 'rgba(14,165,233,', opacity: 0.35 },
            { curve: 'wind',  color: 'rgba(34,197,94,',  opacity: 0.30 },
            { curve: 'solar', color: 'rgba(245,158,11,', opacity: 0.32 },
            { curve: 'storage', color: 'rgba(239,68,68,', opacity: 0.25 }
        ];

        // Seeded RNG for deterministic jitter
        var seed = 73;
        function rng() {
            seed = (seed * 16807) % 2147483647;
            return seed / 2147483647;
        }

        // Pebble grid — hex-packed circles filling the terrain
        var pebbleR = 13;
        var spacingX = 30;
        var spacingY = 26;
        var numCols = Math.ceil(1440 / spacingX) + 1;
        var numRows = Math.ceil(280 / spacingY) + 1;

        var elements = [];

        for (var row = 0; row < numRows && row < 12; row++) {
            for (var col = 0; col < numCols && col < 50; col++) {
                var cx = col * spacingX + (row % 2 ? spacingX / 2 : 0);
                var cy = row * spacingY + spacingY / 2;
                if (cx > 1470 || cy > 285) continue;

                // Jitter position slightly for organic sand feel
                cx += (rng() - 0.5) * 6;
                cy += (rng() - 0.5) * 4;
                var r = pebbleR + (rng() - 0.5) * 3; // vary radius 11.5–14.5

                // Determine which terrain layer this pebble belongs to
                // Check curves top-to-bottom: storage → solar → wind → hydro
                var layerIdx = -1;
                for (var li = layers.length - 1; li >= 0; li--) {
                    var curveY = evalCurve(curves[layers[li].curve], cx);
                    if (cy >= curveY - pebbleR * 0.3) {
                        layerIdx = li;
                        break;
                    }
                }

                if (layerIdx < 0) continue; // above all curves — skip

                var layer = layers[layerIdx];
                // Denser at bottom, lighter near curve boundary
                var curveTop = evalCurve(curves[layer.curve], cx);
                var distFromCurve = cy - curveTop;
                var opScale = Math.min(1.0, 0.5 + distFromCurve / 60);
                var op = (layer.opacity * opScale).toFixed(3);

                // Bottom-up fill: bottom rows animate first (sand settling)
                var begin = ((280 - cy) / 280 * 2.0 + rng() * 0.15).toFixed(2);

                elements.push(
                    '<circle cx="' + cx.toFixed(1) + '" cy="' + cy.toFixed(1) +
                    '" r="' + r.toFixed(1) + '" fill="' + layer.color + op + ')" opacity="0">' +
                    '<animate attributeName="opacity" dur="0.4s" fill="freeze" ' +
                    'begin="' + begin + 's" values="0;1"/>' +
                    '</circle>'
                );
            }
        }

        return elements.join('\n');
    };

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

    // ---- Inject into headers ----
    function injectOverlay() {
        var headers = document.querySelectorAll('.header');
        headers.forEach(function(header) {
            if (header.querySelector('.header-svg-overlay')) return;
            var variant = header.getAttribute('data-header-variant') || 'default';
            // Add CSS variant class
            if (variant !== 'default') {
                header.classList.add('header--' + variant);
            }
            header.insertAdjacentHTML('afterbegin', buildSVG(variant));
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', injectOverlay);
    } else {
        injectOverlay();
    }

    // Expose for dynamic switching (used by variant selector on test pages)
    window._headerVariants = Object.keys(VARIANTS);
    window._switchHeaderVariant = function(variant) {
        var headers = document.querySelectorAll('.header');
        headers.forEach(function(header) {
            // Remove old overlay and variant classes
            var old = header.querySelector('.header-svg-overlay');
            if (old) old.remove();
            window._headerVariants.forEach(function(v) {
                header.classList.remove('header--' + v);
            });
            // Apply new
            if (variant !== 'default') {
                header.classList.add('header--' + variant);
            }
            header.setAttribute('data-header-variant', variant);
            header.insertAdjacentHTML('afterbegin', buildSVG(variant));
        });
    };
})();
