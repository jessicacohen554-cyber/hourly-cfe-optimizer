// ============================================================================
// SHARED BACK BUTTON MODULE - Consistent back navigation bar
// ============================================================================
// Include via: <script src="js/shared-back-button.js"></script>
//
// Usage: Place a placeholder element in your HTML (after the nav):
//   <div id="backBar" data-back-href="about.html" data-back-label="Back to Home"></div>
//
// Attributes:
//   data-back-href  - Link target (default: "about.html")
//   data-back-label - Link text (default: "Back to Home")
//
// This script injects the styled back bar with chevron SVG.
// ============================================================================

(function() {
    'use strict';

    var CHEVRON_SVG = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" ' +
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
        'stroke-linejoin="round"><polyline points="15 18 9 12 15 6"/></svg>';

    function buildBackBar() {
        var el = document.getElementById('backBar');
        if (!el) return;

        var href = el.getAttribute('data-back-href') || 'about.html';
        var label = el.getAttribute('data-back-label') || 'Back to Home';

        el.className = 'back-bar';
        el.innerHTML = '<a href="' + href + '" class="back-bar-link">' +
            CHEVRON_SVG + label + '</a>';
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', buildBackBar);
    } else {
        buildBackBar();
    }
})();
