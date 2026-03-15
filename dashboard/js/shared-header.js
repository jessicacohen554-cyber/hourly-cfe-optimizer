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
// Variants: default | dawn | living | frosted | voltage | pulse | ocean | blueprint
//           topo | constellation | sundial | terrain | hexmosaic | particleflow
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

    // ========== 2. DAWN HORIZON ==========
    VARIANTS['dawn'] = function() {
        return [
            '<defs>',
            '  <linearGradient id="hdr-dawn-amber" x1="0" y1="0" x2="0" y2="1">',
            '    <stop offset="0%" stop-color="rgba(251,146,60,0.22)"/>',
            '    <stop offset="100%" stop-color="rgba(251,146,60,0)"/>',
            '  </linearGradient>',
            '  <linearGradient id="hdr-dawn-teal" x1="0" y1="0" x2="0" y2="1">',
            '    <stop offset="0%" stop-color="rgba(20,184,166,0.14)"/>',
            '    <stop offset="100%" stop-color="rgba(20,184,166,0)"/>',
            '  </linearGradient>',
            '  <linearGradient id="hdr-dawn-blue" x1="0" y1="0" x2="0" y2="1">',
            '    <stop offset="0%" stop-color="rgba(96,165,250,0.12)"/>',
            '    <stop offset="100%" stop-color="rgba(96,165,250,0)"/>',
            '  </linearGradient>',
            '</defs>',
            // Solar curve — dominant, warm amber
            animWave(WAVE_SOLAR, 'hdr-dawn-amber', '0.40', 'rgba(251,146,60,0.35)', '1.8', 15),
            // Wind curve — teal tones
            animWave(WAVE_WIND, 'hdr-dawn-teal', '0.25', 'rgba(20,184,166,0.22)', '1.3', 18),
            // Hydro — soft blue
            animWave(WAVE_HYDRO, 'hdr-dawn-blue', '0.20', 'rgba(96,165,250,0.20)', '1.2', 12),
            // Demand — warm white
            '<path d="' + WAVE_DEMAND.stroke + '" fill="none" stroke="rgba(255,237,213,0.15)" stroke-width="1.5">',
            '  <animate attributeName="d" dur="20s" repeatCount="indefinite" values="' +
                WAVE_DEMAND.stroke + ';' + WAVE_DEMAND.strokeAlt + ';' + WAVE_DEMAND.stroke + '"/>',
            '</path>',
            // Single amber heartbeat — sunrise pulse
            heartbeat(HEARTBEAT_RED, 'rgba(251,146,60,0.20)', '1.5', 3, '0.20', '0.35'),
            // Radial glow at bottom — sunrise effect (via circle)
            '<circle cx="720" cy="270" r="300" fill="rgba(251,146,60,0.06)"/>',
            GRID_DOTS_DARK,
            GRID_LINES_DARK
        ].join('\n');
    };

    // ========== 3. LIVING GRID (Organic/Biomorphic) ==========
    VARIANTS['living'] = function() {
        // More organic curves with thicker strokes and bioluminescent colors
        var organicWave1 = {
            fill: 'M0,190 C100,160 200,200 350,155 C500,110 600,170 750,140 C900,110 1050,165 1200,135 C1300,120 1380,155 1440,145 L1440,280 L0,280 Z',
            fillAlt: 'M0,185 C100,155 200,195 350,160 C500,115 600,175 750,135 C900,115 1050,160 1200,140 C1300,125 1380,150 1440,140 L1440,280 L0,280 Z',
            stroke: 'M0,190 C100,160 200,200 350,155 C500,110 600,170 750,140 C900,110 1050,165 1200,135 C1300,120 1380,155 1440,145',
            strokeAlt: 'M0,185 C100,155 200,195 350,160 C500,115 600,175 750,135 C900,115 1050,160 1200,140 C1300,125 1380,150 1440,140'
        };
        var organicWave2 = {
            fill: 'M0,210 C150,180 300,220 500,175 C700,130 850,190 1000,160 C1150,130 1300,175 1440,165 L1440,280 L0,280 Z',
            fillAlt: 'M0,205 C150,175 300,215 500,180 C700,135 850,195 1000,155 C1150,135 1300,170 1440,160 L1440,280 L0,280 Z',
            stroke: 'M0,210 C150,180 300,220 500,175 C700,130 850,190 1000,160 C1150,130 1300,175 1440,165',
            strokeAlt: 'M0,205 C150,175 300,215 500,180 C700,135 850,195 1000,155 C1150,135 1300,170 1440,160'
        };

        // Large heartbeat — prominent, center of design
        var bigHB = '<path d="M0,140 L160,140 L185,140 L195,136 L205,144 ' +
            'L215,105 L225,180 L238,70 L250,185 L260,100 L272,150 L285,140 ' +
            'L430,140 L470,140 L480,136 L490,144 ' +
            'L500,100 L510,178 L523,65 L535,182 L545,95 L558,152 L570,140 ' +
            'L715,140 L755,140 L765,136 L775,144 ' +
            'L785,108 L795,175 L808,72 L820,180 L830,102 L842,148 L855,140 ' +
            'L1000,140 L1040,140 L1050,136 L1060,144 ' +
            'L1070,102 L1080,180 L1093,68 L1105,184 L1115,98 L1127,150 L1140,140 ' +
            'L1300,140 L1340,140 L1350,136 L1360,144 ' +
            'L1370,110 L1380,172 L1393,78 L1405,178 L1415,105 L1427,148 L1440,140"';

        // Pulse dots along heartbeat peaks
        var pulseDots = [
            '<circle cx="238" cy="70" r="2.5" fill="rgba(239,68,68,0.15)"><animate attributeName="r" dur="3s" repeatCount="indefinite" values="2;4;2"/><animate attributeName="opacity" dur="3s" repeatCount="indefinite" values="0.15;0.40;0.15"/></circle>',
            '<circle cx="523" cy="65" r="2.5" fill="rgba(239,68,68,0.15)"><animate attributeName="r" dur="3s" repeatCount="indefinite" values="2;4;2" begin="0.5s"/><animate attributeName="opacity" dur="3s" repeatCount="indefinite" values="0.15;0.40;0.15" begin="0.5s"/></circle>',
            '<circle cx="808" cy="72" r="2.5" fill="rgba(239,68,68,0.15)"><animate attributeName="r" dur="3s" repeatCount="indefinite" values="2;4;2" begin="1.0s"/><animate attributeName="opacity" dur="3s" repeatCount="indefinite" values="0.15;0.40;0.15" begin="1.0s"/></circle>',
            '<circle cx="1093" cy="68" r="2.5" fill="rgba(239,68,68,0.15)"><animate attributeName="r" dur="3s" repeatCount="indefinite" values="2;4;2" begin="1.5s"/><animate attributeName="opacity" dur="3s" repeatCount="indefinite" values="0.15;0.40;0.15" begin="1.5s"/></circle>',
            '<circle cx="1393" cy="78" r="2.5" fill="rgba(239,68,68,0.15)"><animate attributeName="r" dur="3s" repeatCount="indefinite" values="2;4;2" begin="2.0s"/><animate attributeName="opacity" dur="3s" repeatCount="indefinite" values="0.15;0.40;0.15" begin="2.0s"/></circle>'
        ].join('\n');

        // Bioluminescent floating dots
        var bioDots = [
            '<circle cx="150" cy="60" r="2" fill="rgba(6,182,212,0.08)"><animate attributeName="opacity" dur="4s" repeatCount="indefinite" values="0.05;0.20;0.05"/></circle>',
            '<circle cx="380" cy="90" r="1.5" fill="rgba(16,185,129,0.10)"><animate attributeName="opacity" dur="5s" repeatCount="indefinite" values="0.05;0.18;0.05" begin="1s"/></circle>',
            '<circle cx="550" cy="45" r="2" fill="rgba(6,182,212,0.08)"><animate attributeName="opacity" dur="6s" repeatCount="indefinite" values="0.04;0.16;0.04" begin="2s"/></circle>',
            '<circle cx="820" cy="70" r="1.8" fill="rgba(16,185,129,0.10)"><animate attributeName="opacity" dur="4.5s" repeatCount="indefinite" values="0.06;0.22;0.06" begin="0.5s"/></circle>',
            '<circle cx="1050" cy="55" r="2" fill="rgba(6,182,212,0.08)"><animate attributeName="opacity" dur="7s" repeatCount="indefinite" values="0.05;0.18;0.05" begin="3s"/></circle>',
            '<circle cx="1280" cy="80" r="1.5" fill="rgba(16,185,129,0.10)"><animate attributeName="opacity" dur="5.5s" repeatCount="indefinite" values="0.04;0.20;0.04" begin="1.5s"/></circle>',
            '<circle cx="260" cy="230" r="1.8" fill="rgba(6,182,212,0.06)"><animate attributeName="opacity" dur="6s" repeatCount="indefinite" values="0.04;0.15;0.04" begin="2.5s"/></circle>',
            '<circle cx="680" cy="250" r="2" fill="rgba(16,185,129,0.08)"><animate attributeName="opacity" dur="5s" repeatCount="indefinite" values="0.05;0.17;0.05" begin="0.8s"/></circle>',
            '<circle cx="1150" cy="240" r="1.5" fill="rgba(6,182,212,0.06)"><animate attributeName="opacity" dur="7s" repeatCount="indefinite" values="0.03;0.14;0.03" begin="3.5s"/></circle>'
        ].join('\n');

        return [
            '<defs>',
            '  <linearGradient id="hdr-liv-cyan" x1="0" y1="0" x2="0" y2="1">',
            '    <stop offset="0%" stop-color="rgba(6,182,212,0.18)"/>',
            '    <stop offset="100%" stop-color="rgba(6,182,212,0)"/>',
            '  </linearGradient>',
            '  <linearGradient id="hdr-liv-emerald" x1="0" y1="0" x2="0" y2="1">',
            '    <stop offset="0%" stop-color="rgba(16,185,129,0.14)"/>',
            '    <stop offset="100%" stop-color="rgba(16,185,129,0)"/>',
            '  </linearGradient>',
            '</defs>',
            // Organic curves — thicker, more flowing
            animWave(organicWave1, 'hdr-liv-cyan', '0.35', 'rgba(6,182,212,0.35)', '2.5', 15),
            animWave(organicWave2, 'hdr-liv-emerald', '0.30', 'rgba(16,185,129,0.28)', '2.2', 20),
            // Prominent heartbeat
            heartbeat(bigHB, 'rgba(239,68,68,0.22)', '2.5', 2.5, '0.22', '0.45'),
            pulseDots,
            bioDots,
            // Radial glows behind heartbeat peaks
            '<circle cx="238" cy="100" r="60" fill="rgba(239,68,68,0.04)"/>',
            '<circle cx="808" cy="100" r="60" fill="rgba(239,68,68,0.04)"/>',
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

    // ========== 5. VOLTAGE GRADIENT (Multi-color Bands) ==========
    VARIANTS['voltage'] = function() {
        return [
            '<defs>',
            // Multi-color stroke gradients for curves
            '  <linearGradient id="hdr-volt-curve1" x1="0" y1="0" x2="1" y2="0">',
            '    <stop offset="0%" stop-color="rgba(14,165,233,0.40)"/>',
            '    <stop offset="50%" stop-color="rgba(34,197,94,0.35)"/>',
            '    <stop offset="100%" stop-color="rgba(245,158,11,0.40)"/>',
            '  </linearGradient>',
            '  <linearGradient id="hdr-volt-curve2" x1="0" y1="0" x2="1" y2="0">',
            '    <stop offset="0%" stop-color="rgba(99,102,241,0.35)"/>',
            '    <stop offset="50%" stop-color="rgba(6,182,212,0.30)"/>',
            '    <stop offset="100%" stop-color="rgba(34,197,94,0.35)"/>',
            '  </linearGradient>',
            '  <linearGradient id="hdr-volt-hb" x1="0" y1="0" x2="1" y2="0">',
            '    <stop offset="0%" stop-color="rgba(239,68,68,0.25)"/>',
            '    <stop offset="50%" stop-color="rgba(233,30,99,0.22)"/>',
            '    <stop offset="100%" stop-color="rgba(156,39,176,0.25)"/>',
            '  </linearGradient>',
            '  <linearGradient id="hdr-volt-fill1" x1="0" y1="0" x2="1" y2="0">',
            '    <stop offset="0%" stop-color="rgba(14,165,233,0.12)"/>',
            '    <stop offset="50%" stop-color="rgba(34,197,94,0.10)"/>',
            '    <stop offset="100%" stop-color="rgba(245,158,11,0.12)"/>',
            '  </linearGradient>',
            '  <linearGradient id="hdr-volt-fill2" x1="0" y1="0" x2="1" y2="0">',
            '    <stop offset="0%" stop-color="rgba(99,102,241,0.10)"/>',
            '    <stop offset="50%" stop-color="rgba(6,182,212,0.08)"/>',
            '    <stop offset="100%" stop-color="rgba(34,197,94,0.10)"/>',
            '  </linearGradient>',
            '</defs>',
            // Curve 1 — hydro→wind→solar color shift
            animWave(WAVE_HYDRO, 'hdr-volt-fill1', '0.30', 'url(#hdr-volt-curve1)', '2.0', 12),
            // Curve 2 — nuclear→battery→wind color shift
            animWave(WAVE_WIND, 'hdr-volt-fill2', '0.25', 'url(#hdr-volt-curve2)', '1.5', 18),
            // Solar wave with standard amber
            animWave(WAVE_SOLAR, null, '0', 'rgba(245,158,11,0.20)', '1.2', 15),
            // Demand — white
            '<path d="' + WAVE_DEMAND.stroke + '" fill="none" stroke="rgba(255,255,255,0.10)" stroke-width="1.5">',
            '  <animate attributeName="d" dur="20s" repeatCount="indefinite" values="' +
                WAVE_DEMAND.stroke + ';' + WAVE_DEMAND.strokeAlt + ';' + WAVE_DEMAND.stroke + '"/>',
            '</path>',
            // Gradient heartbeat
            heartbeat(HEARTBEAT_RED, 'url(#hdr-volt-hb)', '1.5', 3, '0.22', '0.38'),
            heartbeat(HEARTBEAT_CYAN, 'rgba(156,39,176,0.12)', '1.0', 4, '0.12', '0.22'),
            // Zone divider lines
            '<line x1="480" y1="0" x2="480" y2="280" stroke="rgba(255,255,255,0.05)" stroke-width="0.5" stroke-dasharray="6 8"/>',
            '<line x1="960" y1="0" x2="960" y2="280" stroke="rgba(255,255,255,0.05)" stroke-width="0.5" stroke-dasharray="6 8"/>',
            GRID_DOTS_DARK,
            GRID_LINES_DARK
        ].join('\n');
    };

    // ========== 6. PULSE MONITOR (Minimal/Clinical) ==========
    VARIANTS['pulse'] = function() {
        // Single bright green heartbeat on near-black with flatline sections
        var greenHB = '<path d="M0,140 L120,140 L180,140 L200,140 L210,138 L218,142 ' +
            'L225,115 L232,170 L240,80 L248,175 L255,108 L262,148 L270,140 ' +
            'L400,140 ' +  // flatline section
            'L500,140 L510,138 L518,142 ' +
            'L525,112 L532,168 L540,75 L548,172 L555,105 L562,150 L570,140 ' +
            'L700,140 ' +  // flatline
            'L800,140 L810,138 L818,142 ' +
            'L825,118 L832,165 L840,82 L848,170 L855,110 L862,146 L870,140 ' +
            'L1000,140 ' +  // flatline
            'L1100,140 L1110,138 L1118,142 ' +
            'L1125,114 L1132,168 L1140,78 L1148,174 L1155,106 L1162,148 L1170,140 ' +
            'L1300,140 ' +  // flatline
            'L1380,140 L1390,138 L1398,142 ' +
            'L1405,120 L1412,162 L1420,88 L1428,168 L1435,112 L1440,140"';

        // Green grid lines
        var greenGrid = [
            '<line x1="0" y1="56" x2="1440" y2="56" stroke="rgba(34,197,94,0.05)" stroke-width="0.5"/>',
            '<line x1="0" y1="84" x2="1440" y2="84" stroke="rgba(34,197,94,0.04)" stroke-width="0.5"/>',
            '<line x1="0" y1="112" x2="1440" y2="112" stroke="rgba(34,197,94,0.05)" stroke-width="0.5"/>',
            '<line x1="0" y1="140" x2="1440" y2="140" stroke="rgba(34,197,94,0.06)" stroke-width="0.5"/>',
            '<line x1="0" y1="168" x2="1440" y2="168" stroke="rgba(34,197,94,0.05)" stroke-width="0.5"/>',
            '<line x1="0" y1="196" x2="1440" y2="196" stroke="rgba(34,197,94,0.04)" stroke-width="0.5"/>',
            '<line x1="0" y1="224" x2="1440" y2="224" stroke="rgba(34,197,94,0.05)" stroke-width="0.5"/>'
        ].join('\n');

        // Dots at peaks
        var peakDots = [
            '<circle cx="240" cy="80" r="2" fill="rgba(34,197,94,0.30)"><animate attributeName="r" dur="2.5s" repeatCount="indefinite" values="2;3.5;2"/><animate attributeName="opacity" dur="2.5s" repeatCount="indefinite" values="0.30;0.70;0.30"/></circle>',
            '<circle cx="540" cy="75" r="2" fill="rgba(34,197,94,0.30)"><animate attributeName="r" dur="2.5s" repeatCount="indefinite" values="2;3.5;2" begin="0.5s"/><animate attributeName="opacity" dur="2.5s" repeatCount="indefinite" values="0.30;0.70;0.30" begin="0.5s"/></circle>',
            '<circle cx="840" cy="82" r="2" fill="rgba(34,197,94,0.30)"><animate attributeName="r" dur="2.5s" repeatCount="indefinite" values="2;3.5;2" begin="1.0s"/><animate attributeName="opacity" dur="2.5s" repeatCount="indefinite" values="0.30;0.70;0.30" begin="1.0s"/></circle>',
            '<circle cx="1140" cy="78" r="2" fill="rgba(34,197,94,0.30)"><animate attributeName="r" dur="2.5s" repeatCount="indefinite" values="2;3.5;2" begin="1.5s"/><animate attributeName="opacity" dur="2.5s" repeatCount="indefinite" values="0.30;0.70;0.30" begin="1.5s"/></circle>',
            '<circle cx="1420" cy="88" r="2" fill="rgba(34,197,94,0.30)"><animate attributeName="r" dur="2.5s" repeatCount="indefinite" values="2;3.5;2" begin="2.0s"/><animate attributeName="opacity" dur="2.5s" repeatCount="indefinite" values="0.30;0.70;0.30" begin="2.0s"/></circle>'
        ].join('\n');

        return [
            greenGrid,
            // Single bright green heartbeat
            heartbeat(greenHB, 'rgba(34,197,94,0.60)', '2.5', 2.5, '0.55', '0.85'),
            peakDots
        ].join('\n');
    };

    // ========== 7. DEEP OCEAN (Teal/Aquatic) ==========
    VARIANTS['ocean'] = function() {
        // Wider, slower ocean-current curves in blue-teal spectrum
        var current1 = {
            fill: 'M0,195 C180,170 360,200 540,180 C720,160 900,190 1080,175 C1200,165 1360,185 1440,178 L1440,280 L0,280 Z',
            fillAlt: 'M0,190 C180,165 360,195 540,185 C720,165 900,185 1080,170 C1200,162 1360,182 1440,175 L1440,280 L0,280 Z',
            stroke: 'M0,195 C180,170 360,200 540,180 C720,160 900,190 1080,175 C1200,165 1360,185 1440,178',
            strokeAlt: 'M0,190 C180,165 360,195 540,185 C720,165 900,185 1080,170 C1200,162 1360,182 1440,175'
        };
        var current2 = {
            fill: 'M0,170 C200,145 400,175 600,150 C800,125 1000,160 1200,140 C1350,128 1420,148 1440,142 L1440,280 L0,280 Z',
            fillAlt: 'M0,165 C200,140 400,170 600,155 C800,130 1000,155 1200,138 C1350,125 1420,145 1440,140 L1440,280 L0,280 Z',
            stroke: 'M0,170 C200,145 400,175 600,150 C800,125 1000,160 1200,140 C1350,128 1420,148 1440,142',
            strokeAlt: 'M0,165 C200,140 400,170 600,155 C800,130 1000,155 1200,138 C1350,125 1420,145 1440,140'
        };
        var current3 = {
            stroke: 'M0,215 C300,200 600,220 900,205 C1100,195 1300,210 1440,205',
            strokeAlt: 'M0,212 C300,198 600,218 900,208 C1100,198 1300,208 1440,202'
        };
        var current4 = {
            stroke: 'M0,240 C250,232 500,245 750,235 C1000,228 1250,240 1440,234',
            strokeAlt: 'M0,238 C250,230 500,242 750,238 C1000,230 1250,238 1440,232'
        };

        // Bioluminescent dots — cyan/teal twinkle
        var bioOcean = [
            '<circle cx="120" cy="55" r="2" fill="rgba(6,182,212,0.10)"><animate attributeName="opacity" dur="5s" repeatCount="indefinite" values="0.05;0.22;0.05"/></circle>',
            '<circle cx="340" cy="85" r="1.5" fill="rgba(20,184,166,0.08)"><animate attributeName="opacity" dur="6s" repeatCount="indefinite" values="0.04;0.18;0.04" begin="1.2s"/></circle>',
            '<circle cx="520" cy="40" r="2.2" fill="rgba(34,211,238,0.06)"><animate attributeName="opacity" dur="7s" repeatCount="indefinite" values="0.03;0.16;0.03" begin="2s"/></circle>',
            '<circle cx="750" cy="70" r="1.8" fill="rgba(6,182,212,0.10)"><animate attributeName="opacity" dur="4.5s" repeatCount="indefinite" values="0.06;0.24;0.06" begin="0.5s"/></circle>',
            '<circle cx="980" cy="50" r="2" fill="rgba(20,184,166,0.08)"><animate attributeName="opacity" dur="5.5s" repeatCount="indefinite" values="0.04;0.20;0.04" begin="3s"/></circle>',
            '<circle cx="1180" cy="75" r="1.5" fill="rgba(34,211,238,0.06)"><animate attributeName="opacity" dur="6.5s" repeatCount="indefinite" values="0.03;0.15;0.03" begin="1.5s"/></circle>',
            '<circle cx="1380" cy="60" r="2" fill="rgba(6,182,212,0.10)"><animate attributeName="opacity" dur="5s" repeatCount="indefinite" values="0.05;0.20;0.05" begin="2.5s"/></circle>',
            '<circle cx="200" cy="240" r="1.8" fill="rgba(20,184,166,0.06)"><animate attributeName="opacity" dur="7s" repeatCount="indefinite" values="0.03;0.14;0.03" begin="3.5s"/></circle>',
            '<circle cx="600" cy="255" r="2" fill="rgba(6,182,212,0.08)"><animate attributeName="opacity" dur="6s" repeatCount="indefinite" values="0.04;0.16;0.04" begin="0.8s"/></circle>',
            '<circle cx="1000" cy="248" r="1.5" fill="rgba(34,211,238,0.06)"><animate attributeName="opacity" dur="5.5s" repeatCount="indefinite" values="0.03;0.12;0.03" begin="4s"/></circle>'
        ].join('\n');

        return [
            '<defs>',
            '  <linearGradient id="hdr-oc-cyan" x1="0" y1="0" x2="0" y2="1">',
            '    <stop offset="0%" stop-color="rgba(6,182,212,0.18)"/>',
            '    <stop offset="100%" stop-color="rgba(6,182,212,0)"/>',
            '  </linearGradient>',
            '  <linearGradient id="hdr-oc-teal" x1="0" y1="0" x2="0" y2="1">',
            '    <stop offset="0%" stop-color="rgba(20,184,166,0.14)"/>',
            '    <stop offset="100%" stop-color="rgba(20,184,166,0)"/>',
            '  </linearGradient>',
            '</defs>',
            // Layer 1 — cyan
            animWave(current1, 'hdr-oc-cyan', '0.30', 'rgba(6,182,212,0.25)', '1.8', 20),
            // Layer 2 — teal
            animWave(current2, 'hdr-oc-teal', '0.25', 'rgba(20,184,166,0.22)', '1.5', 25),
            // Layer 3 — light cyan (stroke only)
            '<path d="' + current3.stroke + '" fill="none" stroke="rgba(34,211,238,0.15)" stroke-width="1.2">',
            '  <animate attributeName="d" dur="28s" repeatCount="indefinite" values="' +
                current3.stroke + ';' + current3.strokeAlt + ';' + current3.stroke + '"/>',
            '</path>',
            // Layer 4 — very light (stroke only)
            '<path d="' + current4.stroke + '" fill="none" stroke="rgba(153,246,228,0.10)" stroke-width="1.0">',
            '  <animate attributeName="d" dur="30s" repeatCount="indefinite" values="' +
                current4.stroke + ';' + current4.strokeAlt + ';' + current4.stroke + '"/>',
            '</path>',
            // Warm coral heartbeat — contrast element
            heartbeat(HEARTBEAT_RED, 'rgba(251,113,133,0.18)', '1.5', 3, '0.18', '0.35'),
            bioOcean,
            // Subtle grid lines in teal
            '<line x1="0" y1="70" x2="1440" y2="70" stroke="rgba(20,184,166,0.04)" stroke-width="0.5"/>',
            '<line x1="0" y1="140" x2="1440" y2="140" stroke="rgba(20,184,166,0.05)" stroke-width="0.5"/>',
            '<line x1="0" y1="210" x2="1440" y2="210" stroke="rgba(20,184,166,0.04)" stroke-width="0.5"/>'
        ].join('\n');
    };

    // ========== 8. BLUEPRINT (Technical/Schematic) ==========
    VARIANTS['blueprint'] = function() {
        // Dense grid (graph paper effect)
        var graphGrid = [];
        // Horizontal lines every 28px
        for (var y = 28; y < 280; y += 28) {
            var op = (y === 140) ? '0.08' : '0.05';
            graphGrid.push('<line x1="0" y1="' + y + '" x2="1440" y2="' + y + '" stroke="rgba(148,163,184,' + op + ')" stroke-width="0.5"/>');
        }
        // Vertical lines every ~160px
        for (var x = 160; x < 1440; x += 160) {
            graphGrid.push('<line x1="' + x + '" y1="0" x2="' + x + '" y2="280" stroke="rgba(148,163,184,0.04)" stroke-width="0.5"/>');
        }

        // Measurement markers — small + marks at curve/grid intersections
        var markers = [
            // Hydro wave intersections
            '<g stroke="rgba(96,165,250,0.25)" stroke-width="1" fill="none">',
            '  <line x1="357" y1="167" x2="363" y2="173"/><line x1="363" y1="167" x2="357" y2="173"/>',
            '  <line x1="717" y1="187" x2="723" y2="193"/><line x1="723" y1="187" x2="717" y2="193"/>',
            '  <line x1="1077" y1="167" x2="1083" y2="173"/><line x1="1083" y1="167" x2="1077" y2="173"/>',
            '</g>',
            // Solar wave intersections
            '<g stroke="rgba(251,191,36,0.25)" stroke-width="1" fill="none">',
            '  <line x1="447" y1="137" x2="453" y2="143"/><line x1="453" y1="137" x2="447" y2="143"/>',
            '  <line x1="717" y1="87" x2="723" y2="93"/><line x1="723" y1="87" x2="717" y2="93"/>',
            '  <line x1="1047" y1="162" x2="1053" y2="168"/><line x1="1053" y1="162" x2="1047" y2="168"/>',
            '</g>',
            // Wind wave intersections
            '<g stroke="rgba(74,222,128,0.25)" stroke-width="1" fill="none">',
            '  <line x1="277" y1="157" x2="283" y2="163"/><line x1="283" y1="157" x2="277" y2="163"/>',
            '  <line x1="597" y1="147" x2="603" y2="153"/><line x1="603" y1="147" x2="597" y2="153"/>',
            '  <line x1="957" y1="137" x2="963" y2="143"/><line x1="963" y1="137" x2="957" y2="143"/>',
            '</g>'
        ].join('\n');

        // Annotation text
        var annotations = [
            '<text x="255" y="88" font-family="monospace" font-size="8" fill="rgba(6,182,212,0.30)" text-anchor="end">8,760h</text>',
            '<text x="545" y="68" font-family="monospace" font-size="8" fill="rgba(6,182,212,0.30)" text-anchor="end">7 ISOs</text>',
            '<text x="1150" y="85" font-family="monospace" font-size="8" fill="rgba(6,182,212,0.30)" text-anchor="end">5,832</text>'
        ].join('\n');

        return [
            graphGrid.join('\n'),
            // Dashed energy curves — engineering/schematic style, stroke only
            '<path d="' + WAVE_HYDRO.stroke + '" fill="none" stroke="rgba(96,165,250,0.30)" stroke-width="1.2" stroke-dasharray="8 4">',
            '  <animate attributeName="d" dur="12s" repeatCount="indefinite" values="' +
                WAVE_HYDRO.stroke + ';' + WAVE_HYDRO.strokeAlt + ';' + WAVE_HYDRO.stroke + '"/>',
            '</path>',
            '<path d="' + WAVE_SOLAR.stroke + '" fill="none" stroke="rgba(251,191,36,0.28)" stroke-width="1.0" stroke-dasharray="8 4">',
            '  <animate attributeName="d" dur="15s" repeatCount="indefinite" values="' +
                WAVE_SOLAR.stroke + ';' + WAVE_SOLAR.strokeAlt + ';' + WAVE_SOLAR.stroke + '"/>',
            '</path>',
            '<path d="' + WAVE_WIND.stroke + '" fill="none" stroke="rgba(74,222,128,0.25)" stroke-width="1.0" stroke-dasharray="8 4">',
            '  <animate attributeName="d" dur="18s" repeatCount="indefinite" values="' +
                WAVE_WIND.stroke + ';' + WAVE_WIND.strokeAlt + ';' + WAVE_WIND.stroke + '"/>',
            '</path>',
            // Single cyan heartbeat as oscilloscope trace
            heartbeat(HEARTBEAT_RED, 'rgba(6,182,212,0.25)', '1.2', 3, '0.20', '0.40'),
            markers,
            annotations
        ].join('\n');
    };

    // ========== 9. TOPOGRAPHIC CONTOUR (Optimization Landscape) ==========
    VARIANTS['topo'] = function() {
        // Concentric contour ellipses centered at the optimization "peak"
        var cx = 680, cy = 135;
        var rings = [];
        var radii = [
            { rx: 40, ry: 24, op: 0.22, sw: 1.5, dur: 6, begin: 0 },
            { rx: 100, ry: 55, op: 0.18, sw: 1.3, dur: 6, begin: 0.7 },
            { rx: 170, ry: 85, op: 0.15, sw: 1.1, dur: 6, begin: 1.4 },
            { rx: 250, ry: 115, op: 0.12, sw: 1.0, dur: 6, begin: 2.1 },
            { rx: 340, ry: 145, op: 0.09, sw: 0.9, dur: 6, begin: 2.8 },
            { rx: 450, ry: 175, op: 0.07, sw: 0.8, dur: 6, begin: 3.5 },
            { rx: 580, ry: 210, op: 0.05, sw: 0.7, dur: 6, begin: 4.2 }
        ];
        for (var i = 0; i < radii.length; i++) {
            var r = radii[i];
            var opLow = (r.op * 0.4).toFixed(3);
            var opHigh = r.op.toFixed(3);
            // Alternate colors: teal for even, blue-green for odd
            var color = (i % 2 === 0) ? 'rgba(20,184,166,' : 'rgba(34,197,94,';
            rings.push(
                '<ellipse cx="' + cx + '" cy="' + cy + '" rx="' + r.rx + '" ry="' + r.ry + '"' +
                ' fill="none" stroke="' + color + opHigh + ')" stroke-width="' + r.sw + '">' +
                '<animate attributeName="stroke-opacity" dur="' + r.dur + 's" repeatCount="indefinite"' +
                ' values="' + opLow + ';' + opHigh + ';' + opLow + '" begin="' + r.begin + 's"/>' +
                '</ellipse>'
            );
        }

        // Peak crosshair marker
        var peak = [
            '<line x1="' + (cx - 8) + '" y1="' + cy + '" x2="' + (cx + 8) + '" y2="' + cy + '" stroke="rgba(20,184,166,0.35)" stroke-width="1.2"/>',
            '<line x1="' + cx + '" y1="' + (cy - 8) + '" x2="' + cx + '" y2="' + (cy + 8) + '" stroke="rgba(20,184,166,0.35)" stroke-width="1.2"/>',
            '<circle cx="' + cx + '" cy="' + cy + '" r="3" fill="rgba(20,184,166,0.25)" stroke="rgba(20,184,166,0.40)" stroke-width="1">',
            '  <animate attributeName="r" dur="3s" repeatCount="indefinite" values="3;5;3"/>',
            '  <animate attributeName="opacity" dur="3s" repeatCount="indefinite" values="0.5;1;0.5"/>',
            '</circle>'
        ].join('\n');

        // Elevation shading — subtle fill on innermost contours
        var shading = [
            '<ellipse cx="' + cx + '" cy="' + cy + '" rx="40" ry="24" fill="rgba(20,184,166,0.06)"/>',
            '<ellipse cx="' + cx + '" cy="' + cy + '" rx="100" ry="55" fill="rgba(20,184,166,0.03)"/>'
        ].join('\n');

        // Annotation labels
        var labels = [
            '<text x="' + (cx + 48) + '" y="' + (cy + 4) + '" font-family="monospace" font-size="7" fill="rgba(20,184,166,0.30)">optimum</text>',
            '<text x="' + (cx + 260) + '" y="' + (cy - 80) + '" font-family="monospace" font-size="7" fill="rgba(20,184,166,0.20)">cost</text>',
            '<text x="' + (cx - 300) + '" y="' + (cy + 100) + '" font-family="monospace" font-size="7" fill="rgba(34,197,94,0.20)">CFE %</text>'
        ].join('\n');

        return [
            shading,
            rings.join('\n'),
            peak,
            labels,
            GRID_DOTS_DARK
        ].join('\n');
    };

    // ========== 10. CONSTELLATION NETWORK (Interconnected Grid) ==========
    VARIANTS['constellation'] = function() {
        // Node positions: {x, y, r, color} — sparse, elegant placement
        var nodes = [
            // Solar amber nodes
            { x: 180, y: 65, r: 3.5, c: 'rgba(245,158,11,0.35)', dur: 5, begin: 0 },
            { x: 520, y: 45, r: 3.0, c: 'rgba(245,158,11,0.30)', dur: 6, begin: 1.2 },
            { x: 960, y: 55, r: 3.5, c: 'rgba(245,158,11,0.32)', dur: 5.5, begin: 2.5 },
            { x: 1300, y: 70, r: 2.8, c: 'rgba(245,158,11,0.28)', dur: 6.5, begin: 0.8 },
            // Wind green nodes
            { x: 100, y: 160, r: 3.0, c: 'rgba(34,197,94,0.32)', dur: 5.5, begin: 0.5 },
            { x: 380, y: 130, r: 3.5, c: 'rgba(34,197,94,0.35)', dur: 4.5, begin: 1.8 },
            { x: 700, y: 100, r: 4.0, c: 'rgba(34,197,94,0.38)', dur: 5, begin: 3.0 },
            { x: 1050, y: 145, r: 3.0, c: 'rgba(34,197,94,0.30)', dur: 6, begin: 0.3 },
            { x: 1350, y: 165, r: 3.5, c: 'rgba(34,197,94,0.33)', dur: 5.5, begin: 2.0 },
            // Hydro blue nodes
            { x: 250, y: 220, r: 3.5, c: 'rgba(14,165,233,0.35)', dur: 6, begin: 1.0 },
            { x: 580, y: 200, r: 3.0, c: 'rgba(14,165,233,0.30)', dur: 5, begin: 2.2 },
            { x: 850, y: 230, r: 3.5, c: 'rgba(14,165,233,0.33)', dur: 5.5, begin: 0.7 },
            { x: 1150, y: 210, r: 3.0, c: 'rgba(14,165,233,0.28)', dur: 6.5, begin: 3.2 },
            { x: 1400, y: 240, r: 2.8, c: 'rgba(14,165,233,0.25)', dur: 7, begin: 1.5 },
            // Nuclear indigo nodes
            { x: 60, y: 80, r: 2.5, c: 'rgba(99,102,241,0.28)', dur: 7, begin: 0.2 },
            { x: 450, y: 250, r: 3.0, c: 'rgba(99,102,241,0.30)', dur: 6, begin: 2.8 },
            { x: 780, y: 170, r: 3.5, c: 'rgba(99,102,241,0.33)', dur: 5, begin: 1.4 },
            { x: 1220, y: 120, r: 2.8, c: 'rgba(99,102,241,0.28)', dur: 6.5, begin: 3.5 }
        ];

        // Build node circles
        var nodesSVG = [];
        for (var i = 0; i < nodes.length; i++) {
            var n = nodes[i];
            nodesSVG.push(
                '<circle cx="' + n.x + '" cy="' + n.y + '" r="' + n.r + '" fill="' + n.c + '">' +
                '<animate attributeName="opacity" dur="' + n.dur + 's" repeatCount="indefinite"' +
                ' values="0.15;0.55;0.15" begin="' + n.begin + 's"/>' +
                '</circle>'
            );
        }

        // Connection lines between nearby nodes (hand-picked pairs for visual balance)
        var connections = [
            [0,5], [5,1], [1,6], [6,2], [2,7], [7,3],  // upper chain
            [14,4], [4,9], [9,10], [10,11], [11,12], [12,13],  // lower chain
            [0,4], [5,9], [1,10], [6,16], [2,17], [16,11],  // vertical links
            [15,10], [14,0]  // cross links
        ];
        var linesSVG = [];
        for (var j = 0; j < connections.length; j++) {
            var a = nodes[connections[j][0]], b = nodes[connections[j][1]];
            var delay = (j * 0.4).toFixed(1);
            linesSVG.push(
                '<line x1="' + a.x + '" y1="' + a.y + '" x2="' + b.x + '" y2="' + b.y + '"' +
                ' stroke="rgba(255,255,255,0.06)" stroke-width="0.6">' +
                '<animate attributeName="stroke-opacity" dur="8s" repeatCount="indefinite"' +
                ' values="0.03;0.14;0.03" begin="' + delay + 's"/>' +
                '</line>'
            );
        }

        return [
            linesSVG.join('\n'),
            nodesSVG.join('\n')
        ].join('\n');
    };

    // ========== 11. RADIAL SUNDIAL (Solar/Temporal) ==========
    VARIANTS['sundial'] = function() {
        var fx = 100, fy = 260;  // focal point (bottom-left)

        // Radial rays
        var rays = [];
        var rayTargets = [
            { x: 300, y: 0, gold: true }, { x: 500, y: 0, gold: true },
            { x: 700, y: 0, gold: true }, { x: 900, y: 0, gold: true },
            { x: 1100, y: 0, gold: true }, { x: 1300, y: 0, gold: true },
            { x: 1440, y: 30, gold: true }, { x: 1440, y: 100, gold: true },
            { x: 1440, y: 170, gold: true }, { x: 1440, y: 240, gold: true },
            // Darker fossil/night rays
            { x: 0, y: 0, gold: false }, { x: 100, y: 0, gold: false },
            { x: 200, y: 0, gold: false }, { x: 1440, y: 270, gold: false }
        ];
        for (var i = 0; i < rayTargets.length; i++) {
            var t = rayTargets[i];
            var color = t.gold ? 'rgba(245,158,11,' : 'rgba(100,116,139,';
            var op = t.gold ? (0.06 + Math.random() * 0.10).toFixed(3) : '0.06';
            rays.push(
                '<line x1="' + fx + '" y1="' + fy + '" x2="' + t.x + '" y2="' + t.y + '"' +
                ' stroke="' + color + op + ')" stroke-width="0.8"/>'
            );
        }

        // Sweeping arc — traces the daily cycle
        var arcRadius = 220;
        // Partial arc path from bottom-left outward
        var arcPath = 'M' + (fx + arcRadius) + ',' + fy +
            ' A' + arcRadius + ',' + arcRadius + ' 0 0 0 ' + fx + ',' + (fy - arcRadius);
        var arcLen = Math.PI * arcRadius / 2;  // quarter circle
        var dashLen = arcLen * 0.15;

        var arc = [
            '<path d="' + arcPath + '" fill="none" stroke="rgba(245,158,11,0.40)" stroke-width="2"' +
            ' stroke-dasharray="' + dashLen.toFixed(0) + ' ' + (arcLen - dashLen).toFixed(0) + '"' +
            ' stroke-linecap="round">',
            '  <animate attributeName="stroke-dashoffset" from="' + arcLen.toFixed(0) + '" to="0" dur="20s" repeatCount="indefinite"/>',
            '</path>'
        ].join('\n');

        // Hour tick marks along a larger arc
        var ticks = [];
        var tickR = 250;
        for (var h = 0; h < 8; h++) {
            var angle = -Math.PI / 2 + (h / 7) * (Math.PI / 2);
            var x1 = fx + tickR * Math.cos(angle);
            var y1 = fy + tickR * Math.sin(angle);
            var x2 = fx + (tickR + 8) * Math.cos(angle);
            var y2 = fy + (tickR + 8) * Math.sin(angle);
            ticks.push(
                '<line x1="' + x1.toFixed(1) + '" y1="' + y1.toFixed(1) + '"' +
                ' x2="' + x2.toFixed(1) + '" y2="' + y2.toFixed(1) + '"' +
                ' stroke="rgba(245,158,11,0.15)" stroke-width="1"/>'
            );
        }

        // Warm focal glow
        var glow = [
            '<circle cx="' + fx + '" cy="' + fy + '" r="60" fill="rgba(245,158,11,0.06)"/>',
            '<circle cx="' + fx + '" cy="' + fy + '" r="25" fill="rgba(245,158,11,0.10)">',
            '  <animate attributeName="r" dur="4s" repeatCount="indefinite" values="25;30;25"/>',
            '  <animate attributeName="opacity" dur="4s" repeatCount="indefinite" values="0.8;1;0.8"/>',
            '</circle>'
        ].join('\n');

        // 8,760 hour label
        var label = '<text x="' + (fx + 265) + '" y="' + (fy - 8) + '" font-family="monospace" font-size="7" fill="rgba(245,158,11,0.22)">8,760h</text>';

        return [
            rays.join('\n'),
            ticks.join('\n'),
            arc,
            glow,
            label
        ].join('\n');
    };

    // ========== 12. STACKED TERRAIN (Joy Division / Dispatch Art) ==========
    VARIANTS['terrain'] = function() {
        // Custom stacked wave paths — each layer goes to bottom (y=280)
        // Layer 1: Hydro (bottom, blue) — gentle
        var hydro = {
            fill: 'M0,245 C120,238 240,232 360,236 C480,240 600,248 720,242 C840,236 960,230 1080,235 C1200,240 1320,246 1440,242 L1440,280 L0,280 Z',
            fillAlt: 'M0,242 C120,235 240,235 360,238 C480,241 600,245 720,240 C840,234 960,233 1080,237 C1200,241 1320,244 1440,240 L1440,280 L0,280 Z',
            stroke: 'M0,245 C120,238 240,232 360,236 C480,240 600,248 720,242 C840,236 960,230 1080,235 C1200,240 1320,246 1440,242',
            strokeAlt: 'M0,242 C120,235 240,235 360,238 C480,241 600,245 720,240 C840,234 960,233 1080,237 C1200,241 1320,244 1440,240'
        };
        // Layer 2: Wind (green) — more variation
        var wind = {
            fill: 'M0,215 C80,205 160,220 280,200 C400,185 520,210 640,195 C760,180 880,200 1000,188 C1120,175 1240,195 1360,190 L1440,195 L1440,280 L0,280 Z',
            fillAlt: 'M0,212 C80,200 160,215 280,198 C400,188 520,205 640,192 C760,178 880,196 1000,185 C1120,178 1240,192 1360,188 L1440,192 L1440,280 L0,280 Z',
            stroke: 'M0,215 C80,205 160,220 280,200 C400,185 520,210 640,195 C760,180 880,200 1000,188 C1120,175 1240,195 1360,190 L1440,195',
            strokeAlt: 'M0,212 C80,200 160,215 280,198 C400,188 520,205 640,192 C760,178 880,196 1000,185 C1120,178 1240,192 1360,188 L1440,192'
        };
        // Layer 3: Solar (amber) — bell curve shape (midday peak)
        var solar = {
            fill: 'M0,210 C180,200 300,170 450,145 C600,125 720,120 900,140 C1050,158 1200,185 1350,200 L1440,205 L1440,280 L0,280 Z',
            fillAlt: 'M0,208 C180,198 300,168 450,148 C600,128 720,122 900,138 C1050,155 1200,182 1350,198 L1440,203 L1440,280 L0,280 Z',
            stroke: 'M0,210 C180,200 300,170 450,145 C600,125 720,120 900,140 C1050,158 1200,185 1350,200 L1440,205',
            strokeAlt: 'M0,208 C180,198 300,168 450,148 C600,128 720,122 900,138 C1050,155 1200,182 1350,198 L1440,203'
        };
        // Layer 4: Storage (red) — thin band, fills evening gaps
        var storage = {
            fill: 'M0,205 C180,198 350,192 500,185 C650,178 780,175 920,178 C1060,182 1200,190 1350,196 L1440,200 L1440,280 L0,280 Z',
            fillAlt: 'M0,203 C180,196 350,190 500,183 C650,176 780,173 920,176 C1060,180 1200,188 1350,194 L1440,198 L1440,280 L0,280 Z',
            stroke: 'M0,205 C180,198 350,192 500,185 C650,178 780,175 920,178 C1060,182 1200,190 1350,196 L1440,200',
            strokeAlt: 'M0,203 C180,196 350,190 500,183 C650,176 780,173 920,176 C1060,180 1200,188 1350,194 L1440,198'
        };
        // Demand line — above the stack, visible gap
        var demand = {
            stroke: 'M0,155 C240,148 480,160 720,152 C960,145 1200,155 1440,150',
            strokeAlt: 'M0,153 C240,146 480,158 720,150 C960,143 1200,153 1440,148'
        };

        return [
            '<defs>',
            '  <linearGradient id="hdr-ter-hydro" x1="0" y1="0" x2="0" y2="1">',
            '    <stop offset="0%" stop-color="rgba(14,165,233,0.35)"/>',
            '    <stop offset="100%" stop-color="rgba(14,165,233,0.05)"/>',
            '  </linearGradient>',
            '  <linearGradient id="hdr-ter-wind" x1="0" y1="0" x2="0" y2="1">',
            '    <stop offset="0%" stop-color="rgba(34,197,94,0.30)"/>',
            '    <stop offset="100%" stop-color="rgba(34,197,94,0.05)"/>',
            '  </linearGradient>',
            '  <linearGradient id="hdr-ter-solar" x1="0" y1="0" x2="0" y2="1">',
            '    <stop offset="0%" stop-color="rgba(245,158,11,0.35)"/>',
            '    <stop offset="100%" stop-color="rgba(245,158,11,0.05)"/>',
            '  </linearGradient>',
            '  <linearGradient id="hdr-ter-storage" x1="0" y1="0" x2="0" y2="1">',
            '    <stop offset="0%" stop-color="rgba(239,68,68,0.25)"/>',
            '    <stop offset="100%" stop-color="rgba(239,68,68,0.03)"/>',
            '  </linearGradient>',
            '</defs>',
            // Layers painted back-to-front: hydro (bottom) → wind → solar → storage (top)
            animWave(hydro, 'hdr-ter-hydro', '0.40', 'rgba(14,165,233,0.45)', '1.5', 18),
            animWave(wind, 'hdr-ter-wind', '0.35', 'rgba(34,197,94,0.40)', '1.2', 15),
            animWave(solar, 'hdr-ter-solar', '0.35', 'rgba(245,158,11,0.45)', '1.2', 12),
            animWave(storage, 'hdr-ter-storage', '0.30', 'rgba(239,68,68,0.35)', '1.0', 16),
            // Demand line — white, prominent
            '<path d="' + demand.stroke + '" fill="none" stroke="rgba(255,255,255,0.50)" stroke-width="2" stroke-dasharray="6 3">',
            '  <animate attributeName="d" dur="20s" repeatCount="indefinite" values="' +
                demand.stroke + ';' + demand.strokeAlt + ';' + demand.stroke + '"/>',
            '</path>',
            // "Gap" label
            '<text x="720" y="138" font-family="monospace" font-size="8" fill="rgba(255,255,255,0.25)" text-anchor="middle">the gap</text>'
        ].join('\n');
    };

    // ========== 13. HEXAGONAL MOSAIC (Tessellated Heat Map) ==========
    VARIANTS['hexmosaic'] = function() {
        var hexR = 28;
        var hexW = hexR * 1.732;  // ~48.5
        var hexH = hexR * 1.5;    // 42
        var cols = Math.ceil(1440 / hexW) + 1;
        var rows = Math.ceil(280 / hexH) + 1;

        // Resource color palette for hexes
        var colors = [
            'rgba(14,165,233,',   // hydro blue
            'rgba(34,197,94,',    // wind green
            'rgba(245,158,11,',   // solar amber
            'rgba(99,102,241,',   // nuclear indigo
            'rgba(6,182,212,',    // cyan/teal
            'rgba(239,68,68,'     // storage red
        ];

        // Simple seeded pseudo-random for deterministic layout
        var seed = 42;
        function rng() {
            seed = (seed * 16807 + 0) % 2147483647;
            return seed / 2147483647;
        }

        // Bright cluster centers (column, row)
        var clusters = [[4,2], [8,3], [12,2], [6,4], [10,1]];

        function hexPath(cx, cy) {
            var pts = [];
            for (var a = 0; a < 6; a++) {
                var angle = Math.PI / 6 + a * Math.PI / 3;  // pointy-top
                pts.push((cx + hexR * Math.cos(angle)).toFixed(1) + ',' + (cy + hexR * Math.sin(angle)).toFixed(1));
            }
            return 'M' + pts.join(' L') + ' Z';
        }

        function isNearCluster(col, row) {
            for (var c = 0; c < clusters.length; c++) {
                var dx = col - clusters[c][0], dy = row - clusters[c][1];
                if (dx * dx + dy * dy <= 3) return true;
            }
            return false;
        }

        var hexes = [];
        for (var row = 0; row < rows && row < 8; row++) {
            for (var col = 0; col < cols && col < 32; col++) {
                var cx = col * hexW + (row % 2 ? hexW / 2 : 0);
                var cy = row * hexH + hexH / 2;
                if (cx > 1480 || cy > 300) continue;

                var colorIdx = Math.floor(rng() * colors.length);
                var nearCluster = isNearCluster(col, row);
                var baseOp = nearCluster ? (0.08 + rng() * 0.12) : (0.03 + rng() * 0.06);
                var opLow = (baseOp * 0.3).toFixed(3);
                var opHigh = baseOp.toFixed(3);
                var dur = (5 + rng() * 3).toFixed(1);
                var begin = (col * 0.25 + row * 0.3).toFixed(1);

                hexes.push(
                    '<path d="' + hexPath(cx, cy) + '" fill="' + colors[colorIdx] + opHigh + ')" stroke="' + colors[colorIdx] + (baseOp * 0.5).toFixed(3) + ')" stroke-width="0.5">' +
                    '<animate attributeName="opacity" dur="' + dur + 's" repeatCount="indefinite"' +
                    ' values="' + opLow + ';' + opHigh + ';' + opLow + '" begin="' + begin + 's"/>' +
                    '</path>'
                );
            }
        }

        return hexes.join('\n');
    };

    // ========== 14. PARTICLE FLOW (Energy Streams Converging) ==========
    VARIANTS['particleflow'] = function() {
        // Guide paths (invisible) — energy streams converging toward center
        var centerX = 720, centerY = 160;
        var paths = [
            // Solar — from top-center, arcing down to center
            { id: 'hdr-pf-solar', d: 'M720,0 C720,40 680,80 700,120 C720,150 720,155 720,160', color: 'rgba(245,158,11,', dur: 8 },
            // Wind — from left edge, curving to center
            { id: 'hdr-pf-wind', d: 'M0,100 C120,95 280,110 420,120 C560,130 650,145 720,160', color: 'rgba(34,197,94,', dur: 10 },
            // Hydro — from bottom-left, arcing up to center
            { id: 'hdr-pf-hydro', d: 'M100,280 C180,260 300,220 440,200 C580,180 660,168 720,160', color: 'rgba(14,165,233,', dur: 9 },
            // Wind2 — from right edge, curving to center
            { id: 'hdr-pf-wind2', d: 'M1440,120 C1320,115 1160,125 1020,135 C880,145 790,152 720,160', color: 'rgba(34,197,94,', dur: 11 }
        ];

        var defs = ['<defs>'];
        for (var i = 0; i < paths.length; i++) {
            defs.push('  <path id="' + paths[i].id + '" d="' + paths[i].d + '" fill="none" stroke="none"/>');
        }
        defs.push('</defs>');

        // Subtle stream trail lines (visible but faint)
        var trails = [];
        for (var t = 0; t < paths.length; t++) {
            trails.push(
                '<path d="' + paths[t].d + '" fill="none" stroke="' + paths[t].color + '0.06)" stroke-width="1"/>'
            );
        }

        // Particles — 6 per stream path
        var particles = [];
        for (var p = 0; p < paths.length; p++) {
            var stream = paths[p];
            var particlesPerStream = 6;
            for (var j = 0; j < particlesPerStream; j++) {
                var pSize = 1.5 + (j % 3) * 0.5;
                var pOpacity = (0.20 + (j % 3) * 0.10).toFixed(2);
                var pDelay = (j * (stream.dur / particlesPerStream)).toFixed(1);
                particles.push(
                    '<circle r="' + pSize + '" fill="' + stream.color + pOpacity + ')">',
                    '  <animateMotion dur="' + stream.dur + 's" repeatCount="indefinite" begin="' + pDelay + 's">',
                    '    <mpath href="#' + stream.id + '"/>',
                    '  </animateMotion>',
                    '</circle>'
                );
            }
        }

        // Convergence glow at center
        var glow = [
            '<circle cx="' + centerX + '" cy="' + centerY + '" r="30" fill="rgba(255,255,255,0.03)"/>',
            '<circle cx="' + centerX + '" cy="' + centerY + '" r="12" fill="rgba(255,255,255,0.05)">',
            '  <animate attributeName="r" dur="3s" repeatCount="indefinite" values="12;16;12"/>',
            '  <animate attributeName="opacity" dur="3s" repeatCount="indefinite" values="0.6;1;0.6"/>',
            '</circle>'
        ].join('\n');

        return [
            defs.join('\n'),
            trails.join('\n'),
            particles.join('\n'),
            glow
        ].join('\n');
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
