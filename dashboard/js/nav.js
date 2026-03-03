// ============================================================================
// SHARED NAVIGATION MODULE — Single source of truth for site navigation
// ============================================================================
// Include via: <script src="js/nav.js"></script>
// Place a <nav id="topNav"></nav> element in HTML, or this script will
// insert the nav as the first child of <body>.
//
// Supports:
//   - Flat top-level links
//   - Standard dropdowns (single column)
//   - Mega-menu dropdowns (multi-column with group headers)
// ============================================================================

(function() {
    'use strict';

    // --- Inject dropdown + mega-menu CSS ---
    var style = document.createElement('style');
    style.textContent = [
        '/* ===== nav.js dropdown & mega-menu ===== */',

        '/* --- Dropdown base --- */',
        '.nav-dropdown { position: relative; }',
        '.nav-dropdown-toggle {',
        '  background: none; border: none; cursor: pointer;',
        '  color: rgba(255,255,255,0.75);',
        '  font-family: var(--font-heading, "Space Grotesk", "Barlow Semi Condensed", sans-serif);',
        '  font-size: 0.85rem; font-weight: 600; letter-spacing: 0.04em;',
        '  padding: 14px 18px; white-space: nowrap;',
        '  transition: color 0.2s, background 0.2s;',
        '  display: flex; align-items: center; gap: 5px;',
        '}',
        '.nav-dropdown-toggle:hover { color: #ffffff; background: rgba(255,255,255,0.08); }',
        '.nav-dropdown.dropdown-active > .nav-dropdown-toggle {',
        '  color: #38bdf8; background: rgba(56,189,248,0.1);',
        '}',
        '.dropdown-arrow { transition: transform 0.2s; }',
        '.nav-dropdown-toggle[aria-expanded="true"] .dropdown-arrow { transform: rotate(180deg); }',
        '',

        '/* --- Standard dropdown (single column) --- */',
        '@media (min-width: 769px) {',
        '  .nav-dropdown-menu {',
        '    display: none; position: absolute; top: 100%; left: 0;',
        '    background: #0B1120; border: 1px solid rgba(255,255,255,0.1);',
        '    border-radius: 8px; min-width: 210px; padding: 6px 0;',
        '    box-shadow: 0 12px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(56,189,248,0.05);',
        '    z-index: 1000;',
        '    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);',
        '  }',
        '  .nav-dropdown:hover > .nav-dropdown-menu { display: block; }',
        '  .nav-dropdown:hover > .nav-mega-menu { display: grid; }',
        '  .nav-dropdown-menu a {',
        '    display: block; padding: 10px 18px; border-bottom: none;',
        '    font-size: 0.82rem; color: rgba(255,255,255,0.7);',
        '    transition: color 0.15s, background 0.15s, padding-left 0.15s;',
        '  }',
        '  .nav-dropdown-menu a:hover {',
        '    background: rgba(56,189,248,0.08); color: #ffffff;',
        '    padding-left: 22px;',
        '  }',
        '  .nav-dropdown-menu a.nav-active {',
        '    background: rgba(56,189,248,0.12); border-bottom: none;',
        '    border-left: 3px solid #38bdf8; color: #38bdf8;',
        '    padding-left: 15px;',
        '  }',
        '}',
        '',

        '/* --- Mega-menu (multi-column) --- */',
        '@media (min-width: 769px) {',
        '  .nav-mega-menu {',
        '    display: none; position: absolute; top: 100%; left: 50%;',
        '    transform: translateX(-50%);',
        '    background: #0B1120; border: 1px solid rgba(255,255,255,0.1);',
        '    border-radius: 10px; padding: 20px 8px 16px;',
        '    box-shadow: 0 16px 48px rgba(0,0,0,0.5), 0 0 0 1px rgba(56,189,248,0.06);',
        '    z-index: 1000;',
        '    grid-template-columns: repeat(var(--mega-cols, 3), 1fr);',
        '    gap: 0;',
        '    min-width: 580px;',
        '    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);',
        '  }',
        '  .nav-mega-menu::before {',
        '    content: ""; position: absolute; top: 0; left: 20px; right: 20px;',
        '    height: 1px;',
        '    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.3), transparent);',
        '  }',
        '  .mega-col {',
        '    padding: 0 16px;',
        '    border-right: 1px solid rgba(255,255,255,0.06);',
        '  }',
        '  .mega-col:last-child { border-right: none; }',
        '  .mega-col-header {',
        '    font-size: 0.65rem; font-weight: 700; text-transform: uppercase;',
        '    letter-spacing: 0.1em; color: rgba(56,189,248,0.6);',
        '    padding: 0 12px 8px; margin-bottom: 4px;',
        '    border-bottom: 1px solid rgba(255,255,255,0.06);',
        '    font-family: var(--font-heading, "Space Grotesk", sans-serif);',
        '  }',
        '  .mega-col a {',
        '    display: block; padding: 8px 12px; border-radius: 6px;',
        '    font-size: 0.8rem; color: rgba(255,255,255,0.7);',
        '    transition: color 0.15s, background 0.15s;',
        '    text-decoration: none; white-space: nowrap;',
        '  }',
        '  .mega-col a:hover {',
        '    background: rgba(56,189,248,0.1); color: #ffffff;',
        '  }',
        '  .mega-col a.nav-active {',
        '    background: rgba(56,189,248,0.15); color: #38bdf8;',
        '    box-shadow: inset 3px 0 0 #38bdf8;',
        '  }',
        '  .mega-col .mega-desc {',
        '    display: block; font-size: 0.68rem; color: rgba(255,255,255,0.35);',
        '    margin-top: 2px; font-weight: 400;',
        '  }',
        '}',
        '',

        '/* --- Separator/group label (standard dropdown) --- */',
        '.nav-separator {',
        '  padding: 8px 18px 4px; margin-top: 4px;',
        '  border-top: 1px solid rgba(255,255,255,0.1);',
        '}',
        '.nav-separator span {',
        '  font-size: 0.68rem; font-weight: 700; text-transform: uppercase;',
        '  letter-spacing: 0.08em; color: rgba(255,255,255,0.4);',
        '}',
        '',

        '/* --- Mobile: collapse everything --- */',
        '@media (max-width: 768px) {',
        '  .top-nav { flex-wrap: wrap; padding: 0 12px; justify-content: space-between; }',
        '  .top-nav-inner {',
        '    display: none; flex-direction: column; width: 100%; padding-bottom: 8px;',
        '  }',
        '  .top-nav-inner.nav-open { display: flex; }',
        '  .top-nav a {',
        '    display: block; padding: 12px 16px; border-bottom: none;',
        '    border-left: 3px solid transparent; width: 100%; text-align: left;',
        '  }',
        '  .top-nav a.nav-active {',
        '    border-left-color: #0EA5E9; border-bottom: none;',
        '    background: rgba(14,165,233,0.08);',
        '  }',
        '  .nav-hamburger { display: flex; }',
        '  .nav-dropdown { width: 100%; }',
        '  .nav-dropdown-toggle {',
        '    width: 100%; text-align: left; padding: 12px 16px;',
        '    border-left: 3px solid transparent;',
        '  }',
        '  .nav-dropdown.dropdown-active > .nav-dropdown-toggle {',
        '    border-left-color: #38bdf8;',
        '  }',

        '  /* Standard dropdown mobile */',
        '  .nav-dropdown-menu { display: none; padding-left: 16px; }',
        '  .nav-dropdown-menu a {',
        '    display: block; padding: 10px 16px 10px 24px; font-size: 0.82rem;',
        '    border-left: 3px solid transparent;',
        '  }',
        '  .nav-dropdown-menu a.nav-active {',
        '    border-left-color: #38bdf8; border-bottom: none;',
        '    background: rgba(56,189,248,0.1);',
        '  }',

        '  /* Mega-menu mobile: collapse to grouped list */',
        '  .nav-mega-menu {',
        '    display: none; padding-left: 8px;',
        '  }',
        '  .mega-col {',
        '    padding: 0; border-right: none;',
        '  }',
        '  .mega-col-header {',
        '    font-size: 0.65rem; font-weight: 700; text-transform: uppercase;',
        '    letter-spacing: 0.08em; color: rgba(255,255,255,0.4);',
        '    padding: 10px 16px 4px;',
        '    font-family: var(--font-heading, "Space Grotesk", sans-serif);',
        '  }',
        '  .mega-col a {',
        '    display: block; padding: 10px 16px 10px 28px; font-size: 0.82rem;',
        '    color: rgba(255,255,255,0.75); text-decoration: none;',
        '    border-left: 3px solid transparent;',
        '  }',
        '  .mega-col a.nav-active {',
        '    border-left-color: #38bdf8;',
        '    background: rgba(56,189,248,0.1);',
        '    color: #38bdf8;',
        '  }',
        '  .mega-col .mega-desc { display: none; }',
        '}'
    ].join('\n');
    document.head.appendChild(style);

    // --- Navigation Structure ---
    // Top-level: 4 items (Home, The Grid, Explore mega-menu, Research dropdown)
    //
    // "Explore" is a mega-menu with 3 columns:
    //   Economics | Markets & Grid | Procurement & Strategy
    //
    // "Research" is a standard dropdown.
    //
    // Pages archived: procurement_comparison.html (Legacy — superseded)
    // Pages merged: pipeline_map.html folded into pipeline.html
    // Fleet Survival moved into Markets & Grid column
    const NAV_ITEMS = [
        { label: 'Home', href: 'index.html' },
        { label: 'The Grid', href: 'dashboard.html' },
        {
            label: 'Explore',
            mega: true,
            columns: [
                {
                    header: 'Economics',
                    items: [
                        { label: 'CO\u2082 Abatement', href: 'abatement_dashboard.html', desc: 'Marginal abatement cost curves' },
                        { label: 'Clean Firm Case', href: 'clean_firm_case.html', desc: 'The role of firm clean power' },
                        { label: 'Cost to Replace', href: 'cost_to_replace.html', desc: 'Replacement cost analysis' },
                        { label: 'New Build Analysis', href: 'new_build_analysis.html', desc: 'Greenfield resource economics' },
                        { label: 'Wright\u2019s Law', href: 'wrights_law.html', desc: 'Learning curves & cost trajectories' }
                    ]
                },
                {
                    header: 'Markets & Grid',
                    items: [
                        { label: 'Wholesale Prices', href: 'lmp_trends.html', desc: 'Synthetic LMP analysis' },
                        { label: 'Fossil Fuel Analysis', href: 'fossil_fuel_deepdive.html', desc: 'Gas, coal & oil dynamics' },
                        { label: 'Storage Analysis', href: 'storage_analysis.html', desc: 'Battery & LDES dispatch' },
                        { label: 'Fleet Survival', href: 'fleet_survival.html', desc: 'Generator fleet transitions' }
                    ]
                },
                {
                    header: 'Procurement & Strategy',
                    items: [
                        { label: 'Carbon Accounting', href: 'consequential_accounting.html', desc: 'Consequential vs. attributional' },
                        { label: 'Scenario Comparison', href: 'scenario_comparison.html', desc: 'Procurement pathway analysis' },
                        { label: 'Procurement Strategies', href: 'procurement_strategies.html', desc: 'Corporate strategy comparison' },
                        { label: 'Failure Modes', href: 'consequential_vacuum.html', desc: 'Scaling failure analysis' }
                    ]
                }
            ]
        },
        {
            label: 'Research',
            children: [
                { label: 'Research Paper', href: 'research_paper.html' },
                { label: 'Methodology', href: 'optimizer_methodology.html' },
                { label: 'Policy Context', href: 'policy_context.html' },
                { label: 'About the Model', href: 'pipeline.html' },
                { label: 'About', href: 'about.html' },
                { label: 'Reference', href: 'reference.html' }
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
        const normalized = href.replace(/^\.\.\//, '').replace(/^\.\//, '');
        return current === normalized || current === href;
    }

    // Check if any child in a dropdown matches the current page
    function hasActiveChild(children) {
        return children.some(function(child) { return child.href && isActive(child.href); });
    }

    // Check if any item in a mega-menu matches the current page
    function hasMegaActiveChild(columns) {
        return columns.some(function(col) {
            return col.items.some(function(item) { return isActive(item.href); });
        });
    }

    // Build a single nav link
    function buildLink(item) {
        var cls = isActive(item.href) ? ' class="nav-active"' : '';
        return '<a href="' + item.href + '"' + cls + '>' + item.label + '</a>';
    }

    // Build a standard dropdown menu (single column)
    function buildDropdown(item) {
        var activeClass = hasActiveChild(item.children) ? ' dropdown-active' : '';
        var html = '<div class="nav-dropdown' + activeClass + '">';
        html += '<button class="nav-dropdown-toggle" aria-expanded="false">';
        html += item.label + ' ' + DROPDOWN_ARROW + '</button>';
        html += '<div class="nav-dropdown-menu">';
        item.children.forEach(function(child) {
            if (child.separator) {
                html += '<div class="nav-separator"><span>' + child.label + '</span></div>';
            } else {
                html += buildLink(child);
            }
        });
        html += '</div></div>';
        return html;
    }

    // Build a mega-menu dropdown (multi-column with group headers)
    function buildMegaMenu(item) {
        var activeClass = hasMegaActiveChild(item.columns) ? ' dropdown-active' : '';
        var colCount = item.columns.length;
        var html = '<div class="nav-dropdown' + activeClass + '">';
        html += '<button class="nav-dropdown-toggle" aria-expanded="false">';
        html += item.label + ' ' + DROPDOWN_ARROW + '</button>';
        html += '<div class="nav-mega-menu" style="--mega-cols:' + colCount + '">';

        item.columns.forEach(function(col) {
            html += '<div class="mega-col">';
            html += '<div class="mega-col-header">' + col.header + '</div>';
            col.items.forEach(function(link) {
                var cls = isActive(link.href) ? ' class="nav-active"' : '';
                html += '<a href="' + link.href + '"' + cls + '>';
                html += link.label;
                if (link.desc) {
                    html += '<span class="mega-desc">' + link.desc + '</span>';
                }
                html += '</a>';
            });
            html += '</div>';
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
            if (item.mega) {
                html += buildMegaMenu(item);
            } else if (item.children) {
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
        var existing = document.getElementById('topNav');
        if (existing) {
            existing.outerHTML = buildNav();
        } else {
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
                // Find the menu — either .nav-dropdown-menu or .nav-mega-menu
                var menu = dropdown.querySelector('.nav-dropdown-menu') || dropdown.querySelector('.nav-mega-menu');
                var isExpanded = this.getAttribute('aria-expanded') === 'true';

                // Close other dropdowns on mobile
                if (window.innerWidth <= 768) {
                    navLinks.querySelectorAll('.nav-dropdown').forEach(function(d) {
                        if (d !== dropdown) {
                            d.querySelector('.nav-dropdown-toggle').setAttribute('aria-expanded', 'false');
                            var otherMenu = d.querySelector('.nav-dropdown-menu') || d.querySelector('.nav-mega-menu');
                            if (otherMenu) otherMenu.style.display = 'none';
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
                navLinks.querySelectorAll('.nav-dropdown-menu, .nav-mega-menu').forEach(function(menu) {
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
