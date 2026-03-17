// ============================================================================
// SHARED FOOTER MODULE - Single source of truth for site footer
// ============================================================================
// Include via: <script src="js/shared-footer.js"></script>
//
// Usage: Place a placeholder element in your HTML:
//   <footer id="siteFooter" data-footer-note="Page-specific note here"></footer>
//
// This script will inject the standard footer links, note, and bottom-banner.
// If no data-footer-note is provided, a default note is used.
// ============================================================================

(function() {
    'use strict';

    // Canonical footer link set - single source of truth
    var FOOTER_LINKS = [
        { href: 'about.html',                label: 'About' },
        { href: 'dashboard.html',             label: 'Cost Optimizer' },
        { href: 'abatement_dashboard.html',   label: 'CO\u2082 Abatement' },
        { href: 'gen_market_overview.html',    label: 'Market Overview' },
        { href: 'GHG-accounting-overview.html', label: 'Carbon Accounting' },
        { href: 'lmp_trends.html',            label: 'LMP Trends' },
        { href: 'optimizer_methodology.html',   label: 'Methodology' },
        { href: 'archive/research_paper.html',  label: 'Research Paper' }
    ];

    var DEFAULT_NOTE = 'The 8,760 Problem \u2014 Hourly Clean Energy Optimization';

    function buildFooter() {
        var el = document.getElementById('siteFooter');
        if (!el) return;

        var note = el.getAttribute('data-footer-note') || DEFAULT_NOTE;

        // Build footer HTML
        el.className = 'page-footer';
        el.innerHTML =
            '<div class="footer-links">' +
            FOOTER_LINKS.map(function(link) {
                return '<a href="' + link.href + '">' + link.label + '</a>';
            }).join('') +
            '</div>' +
            '<div class="footer-note">' + note + '</div>';

        // Insert bottom-banner after footer
        var banner = document.createElement('div');
        banner.className = 'bottom-banner';
        el.parentNode.insertBefore(banner, el.nextSibling);
    }

    // Run on DOMContentLoaded or immediately if already loaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', buildFooter);
    } else {
        buildFooter();
    }
})();
