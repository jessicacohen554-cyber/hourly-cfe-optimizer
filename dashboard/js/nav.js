// ============================================================================
// SHARED NAVIGATION MODULE — Single source of truth for site navigation
// ============================================================================
// Include via: <script src="js/nav.js"></script>
// Place a <nav id="topNav"></nav> element in HTML, or this script will
// insert the nav as the first child of <body>.
// ============================================================================

(function() {
    'use strict';

    // --- Inject dropdown CSS (supplements existing top-nav styles) ---
    var style = document.createElement('style');
    style.textContent = [
        '/* nav.js dropdown additions */',
        '.nav-dropdown { position: relative; }',
        '.nav-dropdown-toggle {',
        '  background: none; border: none; cursor: pointer;',
        '  color: rgba(255,255,255,0.75);',
        '  font-family: "Franklin Gothic Demi", "Barlow Semi Condensed", "Arial Narrow", sans-serif;',
        '  font-size: 0.85rem; font-weight: 600; letter-spacing: 0.04em;',
        '  padding: 14px 18px; white-space: nowrap;',
        '  transition: color 0.2s, background 0.2s;',
        '  display: flex; align-items: center; gap: 5px;',
        '}',
        '.nav-dropdown-toggle:hover { color: #fff; background: rgba(255,255,255,0.06); }',
        '.nav-dropdown.dropdown-active > .nav-dropdown-toggle {',
        '  color: #fff; background: rgba(14,165,233,0.1);',
        '}',
        '.dropdown-arrow { transition: transform 0.2s; }',
        '.nav-dropdown-toggle[aria-expanded="true"] .dropdown-arrow { transform: rotate(180deg); }',
        '',
        '/* Desktop: hover to open */',
        '@media (min-width: 769px) {',
        '  .nav-dropdown-menu {',
        '    display: none; position: absolute; top: 100%; left: 0;',
        '    background: #1a1a2e; border: 1px solid rgba(255,255,255,0.1);',
        '    border-radius: 6px; min-width: 200px; padding: 4px 0;',
        '    box-shadow: 0 8px 24px rgba(0,0,0,0.4); z-index: 1000;',
        '  }',
        '  .nav-dropdown:hover > .nav-dropdown-menu { display: block; }',
        '  .nav-dropdown-menu a {',
        '    display: block; padding: 10px 18px; border-bottom: none;',
        '    font-size: 0.82rem;',
        '  }',
        '  .nav-dropdown-menu a:hover { background: rgba(255,255,255,0.08); }',
        '  .nav-dropdown-menu a.nav-active {',
        '    background: rgba(14,165,233,0.15); border-bottom: none;',
        '    border-left: 3px solid #0EA5E9;',
        '  }',
        '}',
        '',
        '/* Mobile: expand in-place */',
        '@media (max-width: 768px) {',
        '  .top-nav { flex-wrap: wrap; padding: 0 12px; justify-content: space-between; }',
        '  .top-nav-inner {',
        '    display: none; flex-direction: column; width: 100%; padding-bottom: 8px;',
        '  }',
        '  .top-nav-inner.nav-open { display: flex; }',
        '  .top-nav a {',
        '    padding: 12px 16px; border-bottom: none;',
        '    border-left: 3px solid transparent; width: 100%; text-align: left;',
        '  }',
        '  .top-nav a.nav-active {',
        '    border-left-color: #0EA5E9; border-bottom: none;',
        '    background: rgba(14,165,233,0.12);',
        '  }',
        '  .nav-hamburger { display: flex; }',
        '  .nav-dropdown { width: 100%; }',
        '  .nav-dropdown-toggle {',
        '    width: 100%; text-align: left; padding: 12px 16px;',
        '    border-left: 3px solid transparent;',
        '  }',
        '  .nav-dropdown.dropdown-active > .nav-dropdown-toggle {',
        '    border-left-color: #0EA5E9;',
        '  }',
        '  .nav-dropdown-menu { display: none; padding-left: 16px; }',
        '  .nav-dropdown-menu a {',
        '    padding: 10px 16px; font-size: 0.82rem;',
        '    border-left: 3px solid transparent;',
        '  }',
        '  .nav-dropdown-menu a.nav-active {',
        '    border-left-color: #0EA5E9; border-bottom: none;',
        '    background: rgba(14,165,233,0.12);',
        '  }',
        '}'
    ].join('\n');
    document.head.appendChild(style);

    // --- Navigation Structure ---
    // Flat items appear directly in the nav bar.
    // Items with 'children' create dropdown menus on desktop, expandable on mobile.
    const NAV_ITEMS = [
        { label: 'Home', href: 'index.html' },
        { label: 'Grid Simulation', href: 'dashboard.html' },
        { label: 'Clean Firm Case', href: 'clean_firm_case.html' },
        { label: 'CO\u2082 Abatement', href: 'abatement_dashboard.html' },
        {
            label: 'Research',
            children: [
                { label: 'Research Paper', href: 'research_paper.html' },
                { label: 'Methodology', href: 'optimizer_methodology.html' },
                { label: 'Policy Context', href: 'policy_context.html' },
                { label: 'About', href: 'about.html' }
            ]
        },
        {
            label: 'Generator Analysis',
            children: [
                { label: 'Overview', href: '../power-gen-decarbonization/site/index.html' },
                { label: 'Fleet Analysis', href: '../power-gen-decarbonization/site/fleet-analysis.html' },
                { label: 'Dashboard', href: '../power-gen-decarbonization/site/dashboard.html' },
                { label: 'Targets & Standards', href: '../power-gen-decarbonization/site/targets.html' },
                { label: 'Policy Scenarios', href: '../power-gen-decarbonization/site/policy.html' },
                { label: 'Methodology', href: '../power-gen-decarbonization/site/methodology.html' }
            ]
        }
    ];

    const HAMBURGER_OPEN = '<svg viewBox="0 0 24 24"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>';
    const HAMBURGER_CLOSE = '<svg viewBox="0 0 24 24"><line x1="6" y1="6" x2="18" y2="18"/><line x1="6" y1="18" x2="18" y2="6"/></svg>';
    const DROPDOWN_ARROW = '<svg class="dropdown-arrow" viewBox="0 0 12 8" width="10" height="6"><path d="M1 1l5 5 5-5" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>';

    // Detect current page from URL
    function getCurrentPage() {
        const path = window.location.pathname;
        const filename = path.substring(path.lastIndexOf('/') + 1) || 'index.html';
        return filename;
    }

    // Check if an href matches the current page
    function isActive(href) {
        const current = getCurrentPage();
        // Normalize: strip leading ./ or ../
        const normalized = href.replace(/^\.\.\//, '').replace(/^\.\//, '');
        return current === normalized || current === href;
    }

    // Check if any child in a dropdown matches the current page
    function hasActiveChild(children) {
        return children.some(function(child) { return isActive(child.href); });
    }

    // Build a single nav link
    function buildLink(item) {
        var cls = isActive(item.href) ? ' class="nav-active"' : '';
        return '<a href="' + item.href + '"' + cls + '>' + item.label + '</a>';
    }

    // Build a dropdown menu
    function buildDropdown(item) {
        var activeClass = hasActiveChild(item.children) ? ' dropdown-active' : '';
        var html = '<div class="nav-dropdown' + activeClass + '">';
        html += '<button class="nav-dropdown-toggle" aria-expanded="false">';
        html += item.label + ' ' + DROPDOWN_ARROW + '</button>';
        html += '<div class="nav-dropdown-menu">';
        item.children.forEach(function(child) {
            html += buildLink(child);
        });
        html += '</div></div>';
        return html;
    }

    // Build the full nav HTML
    function buildNav() {
        var html = '<nav class="top-nav" id="topNav">';
        html += '<span class="nav-brand">The 8,760 Problem</span>';
        html += '<button class="nav-hamburger" id="navHamburger" aria-label="Toggle navigation menu">' + HAMBURGER_OPEN + '</button>';
        html += '<div class="top-nav-inner" id="navLinks">';

        NAV_ITEMS.forEach(function(item) {
            if (item.children) {
                html += buildDropdown(item);
            } else {
                html += buildLink(item);
            }
        });

        html += '</div></nav>';
        return html;
    }

    // Inject the nav into the page
    function injectNav() {
        // Look for existing placeholder
        var existing = document.getElementById('topNav');
        if (existing) {
            existing.outerHTML = buildNav();
        } else {
            // Insert at the start of body
            document.body.insertAdjacentHTML('afterbegin', buildNav());
        }
    }

    // Wire up hamburger and dropdown interactions
    function wireInteractions() {
        var hamburger = document.getElementById('navHamburger');
        var navLinks = document.getElementById('navLinks');
        if (!hamburger || !navLinks) return;

        // Hamburger toggle
        hamburger.addEventListener('click', function() {
            navLinks.classList.toggle('nav-open');
            var isOpen = navLinks.classList.contains('nav-open');
            this.innerHTML = isOpen ? HAMBURGER_CLOSE : HAMBURGER_OPEN;
        });

        // Close mobile menu on link click
        navLinks.querySelectorAll('a').forEach(function(link) {
            link.addEventListener('click', function() {
                if (window.innerWidth <= 768) {
                    navLinks.classList.remove('nav-open');
                    hamburger.innerHTML = HAMBURGER_OPEN;
                }
            });
        });

        // Dropdown toggles (mobile: click to expand; desktop: hover handled by CSS)
        navLinks.querySelectorAll('.nav-dropdown-toggle').forEach(function(btn) {
            btn.addEventListener('click', function(e) {
                e.preventDefault();
                var dropdown = this.parentElement;
                var menu = dropdown.querySelector('.nav-dropdown-menu');
                var isExpanded = this.getAttribute('aria-expanded') === 'true';

                // Close other dropdowns on mobile
                if (window.innerWidth <= 768) {
                    navLinks.querySelectorAll('.nav-dropdown').forEach(function(d) {
                        if (d !== dropdown) {
                            d.querySelector('.nav-dropdown-toggle').setAttribute('aria-expanded', 'false');
                            d.querySelector('.nav-dropdown-menu').style.display = 'none';
                        }
                    });
                }

                this.setAttribute('aria-expanded', String(!isExpanded));
                menu.style.display = isExpanded ? 'none' : 'block';
            });
        });

        // Desktop: close dropdowns when clicking outside
        document.addEventListener('click', function(e) {
            if (window.innerWidth > 768 && !e.target.closest('.nav-dropdown')) {
                navLinks.querySelectorAll('.nav-dropdown-menu').forEach(function(menu) {
                    menu.style.display = '';
                });
                navLinks.querySelectorAll('.nav-dropdown-toggle').forEach(function(btn) {
                    btn.setAttribute('aria-expanded', 'false');
                });
            }
        });
    }

    // Initialize
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            injectNav();
            wireInteractions();
        });
    } else {
        injectNav();
        wireInteractions();
    }
})();
