/**
 * ceg-nav.js — Shared Constellation Energy navigation bar
 *
 * Auto-injects a branded nav into any page that includes this script.
 * Set `data-ceg-active` on the <script> tag to highlight the current page.
 *
 * Usage:
 *   <script src="js/ceg-nav.js" data-ceg-active="ccs-map"></script>
 *
 * Active keys: "report", "slides", "ccs-map"
 */
(function () {
    'use strict';

    // Determine which page is active from the script tag's data attribute
    const scriptTag = document.currentScript;
    const activePage = scriptTag ? scriptTag.getAttribute('data-ceg-active') || '' : '';

    const pages = [
        { key: 'report',  label: 'Fleet Report',     href: 'ceg_constellation_report.html', icon: '&#128200;' },
        { key: 'slides',  label: 'Slide Deck',        href: 'ceg_constellation_slides.html', icon: '&#128202;' },
        { key: 'ccs-map', label: 'CCS Map',            href: 'ccs_proximity_map.html',        icon: '&#127758;' },
    ];

    // Build nav HTML
    const linksHTML = pages.map(p => {
        const cls = p.key === activePage ? ' class="active"' : '';
        return `<li><a href="${p.href}"${cls}><span class="ceg-nav-icon">${p.icon}</span>${p.label}</a></li>`;
    }).join('\n');

    const navHTML = `
    <nav class="ceg-hub-nav" id="cegHubNav">
        <div class="ceg-hub-nav-inner">
            <a href="ceg_constellation_report.html" class="ceg-hub-brand">
                <img src="img/ceg_logo.png" alt="Constellation Energy" class="ceg-hub-logo"
                     onerror="this.onerror=null;this.src='../market-simulator/frontend/brand-assets/logo.png';">
                <div class="ceg-hub-brand-text">
                    <div class="ceg-hub-title">Constellation Energy</div>
                    <div class="ceg-hub-subtitle">IPP Climate Transition Analysis</div>
                </div>
            </a>
            <button class="ceg-hub-hamburger" id="cegHamburger" aria-label="Toggle menu">
                <span></span><span></span><span></span>
            </button>
            <ul class="ceg-hub-links" id="cegHubLinks">
                ${linksHTML}
                <li class="ceg-hub-back">
                    <a href="ipp_climate_transition.html">&#8592; All IPPs</a>
                </li>
            </ul>
        </div>
    </nav>
    <div class="ceg-hub-nav-spacer"></div>
    `;

    // Inject at top of body
    document.body.insertAdjacentHTML('afterbegin', navHTML);

    // Hamburger toggle
    const hamburger = document.getElementById('cegHamburger');
    const links = document.getElementById('cegHubLinks');
    if (hamburger && links) {
        hamburger.addEventListener('click', function () {
            const open = links.classList.toggle('open');
            hamburger.classList.toggle('open', open);
        });
        // Close on link click (mobile)
        links.querySelectorAll('a').forEach(a => {
            a.addEventListener('click', () => {
                links.classList.remove('open');
                hamburger.classList.remove('open');
            });
        });
    }
})();
