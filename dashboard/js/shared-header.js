// ============================================================================
// SHARED HEADER — Injects the page banner with SVG waveform/heartbeat overlay
// ============================================================================
// Usage: Include this script AFTER shared.css and nav.js
//   <script src="js/shared-header.js"></script>
//
// Then place a simple header element in your HTML:
//   <div class="header" id="pageHeader">
//       <h1>Page Title</h1>
//       <div class="subtitle">Page description</div>
//       <div class="header-accent"></div>
//   </div>
//
// This script will automatically inject the SVG overlay into the header.
// ============================================================================

(function() {
    'use strict';

    // SVG waveform overlay with energy curve lines and heartbeat/EKG pulse
    var SVG_OVERLAY = [
        '<div class="header-svg-overlay">',
        '<svg viewBox="0 0 1440 280" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">',

        // --- Smooth energy curve lines (flowing sinusoidal waves) ---

        // Wave 1: Wide gentle sine — represents baseload (hydro blue)
        '<path d="M0,200 C120,180 240,160 360,170 C480,180 600,200 720,190 ',
              'C840,180 960,160 1080,170 C1200,180 1320,200 1440,190" ',
              'fill="none" stroke="rgba(14,165,233,0.15)" stroke-width="1.5">',
        '  <animate attributeName="d" dur="12s" repeatCount="indefinite" values="',
            'M0,200 C120,180 240,160 360,170 C480,180 600,200 720,190 C840,180 960,160 1080,170 C1200,180 1320,200 1440,190;',
            'M0,195 C120,175 240,165 360,175 C480,185 600,195 720,185 C840,175 960,165 1080,175 C1200,185 1320,195 1440,185;',
            'M0,200 C120,180 240,160 360,170 C480,180 600,200 720,190 C840,180 960,160 1080,170 C1200,180 1320,200 1440,190',
        '"/>',
        '</path>',

        // Wave 2: Solar generation curve — peaks mid-page (amber)
        '<path d="M0,240 C180,230 300,180 450,140 C600,100 720,90 900,130 ',
              'C1050,165 1200,210 1350,230 L1440,240" ',
              'fill="none" stroke="rgba(245,158,11,0.12)" stroke-width="1.2">',
        '  <animate attributeName="d" dur="15s" repeatCount="indefinite" values="',
            'M0,240 C180,230 300,180 450,140 C600,100 720,90 900,130 C1050,165 1200,210 1350,230 L1440,240;',
            'M0,235 C180,225 300,175 450,145 C600,105 720,95 900,125 C1050,160 1200,205 1350,225 L1440,235;',
            'M0,240 C180,230 300,180 450,140 C600,100 720,90 900,130 C1050,165 1200,210 1350,230 L1440,240',
        '"/>',
        '</path>',

        // Wave 3: Wind variability — irregular undulation (green)
        '<path d="M0,180 C80,165 160,190 280,160 C400,130 480,170 600,150 ',
              'C720,130 840,160 960,140 C1080,120 1200,155 1320,145 L1440,160" ',
              'fill="none" stroke="rgba(34,197,94,0.10)" stroke-width="1.0">',
        '  <animate attributeName="d" dur="18s" repeatCount="indefinite" values="',
            'M0,180 C80,165 160,190 280,160 C400,130 480,170 600,150 C720,130 840,160 960,140 C1080,120 1200,155 1320,145 L1440,160;',
            'M0,175 C80,160 160,185 280,155 C400,135 480,165 600,155 C720,135 840,155 960,135 C1080,125 1200,150 1320,140 L1440,155;',
            'M0,180 C80,165 160,190 280,160 C400,130 480,170 600,150 C720,130 840,160 960,140 C1080,120 1200,155 1320,145 L1440,160',
        '"/>',
        '</path>',

        // Wave 4: Subtle demand baseline — smooth low-amplitude (white)
        '<path d="M0,220 C240,215 480,225 720,218 C960,211 1200,222 1440,216" ',
              'fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="1.8">',
        '  <animate attributeName="d" dur="20s" repeatCount="indefinite" values="',
            'M0,220 C240,215 480,225 720,218 C960,211 1200,222 1440,216;',
            'M0,222 C240,217 480,222 720,215 C960,213 1200,220 1440,218;',
            'M0,220 C240,215 480,225 720,218 C960,211 1200,222 1440,216',
        '"/>',
        '</path>',

        // --- Heartbeat / EKG pulse line (center of banner) ---
        // Sharp peaks represent energy dispatch events
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
              'L1440,140" ',
              'fill="none" stroke="rgba(239,68,68,0.18)" stroke-width="1.2" ',
              'stroke-linecap="round" stroke-linejoin="round">',
        '  <animate attributeName="stroke-opacity" dur="3s" repeatCount="indefinite" ',
        '    values="0.18;0.28;0.18" keyTimes="0;0.5;1"/>',
        '</path>',

        // Second heartbeat line — offset vertically, different phase (cyan)
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
              'L1440,155" ',
              'fill="none" stroke="rgba(56,189,248,0.12)" stroke-width="1.0" ',
              'stroke-linecap="round" stroke-linejoin="round">',
        '  <animate attributeName="stroke-opacity" dur="4s" repeatCount="indefinite" ',
        '    values="0.12;0.20;0.12" keyTimes="0;0.5;1"/>',
        '</path>',

        // --- Subtle grid dots (small circles scattered) ---
        '<circle cx="200" cy="80" r="1" fill="rgba(255,255,255,0.08)"/>',
        '<circle cx="400" cy="60" r="1.2" fill="rgba(255,255,255,0.06)"/>',
        '<circle cx="600" cy="100" r="1" fill="rgba(255,255,255,0.07)"/>',
        '<circle cx="800" cy="50" r="1.5" fill="rgba(255,255,255,0.05)"/>',
        '<circle cx="1000" cy="75" r="1" fill="rgba(255,255,255,0.08)"/>',
        '<circle cx="1200" cy="90" r="1.2" fill="rgba(255,255,255,0.06)"/>',
        '<circle cx="300" cy="250" r="1" fill="rgba(255,255,255,0.05)"/>',
        '<circle cx="700" cy="240" r="1.3" fill="rgba(255,255,255,0.06)"/>',
        '<circle cx="1100" cy="260" r="1" fill="rgba(255,255,255,0.07)"/>',

        '</svg>',
        '</div>'
    ].join('\n');

    function injectOverlay() {
        var headers = document.querySelectorAll('.header');
        headers.forEach(function(header) {
            // Don't double-inject
            if (header.querySelector('.header-svg-overlay')) return;
            // Insert the SVG overlay as the first child
            header.insertAdjacentHTML('afterbegin', SVG_OVERLAY);
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', injectOverlay);
    } else {
        injectOverlay();
    }
})();
