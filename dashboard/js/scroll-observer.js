// ============================================================================
// SHARED SCROLL OBSERVER — Centralized IntersectionObserver for fade-in
// ============================================================================
// Usage: Include this script AFTER shared.css (which defines the animation
// classes). Elements with these classes will automatically animate on scroll:
//
//   .scroll-reveal       — Fade in + slide up (0.7s, spring ease)
//   .scroll-reveal-slow  — Fade in + slide up (0.8s, ease-out-quart)
//   .story-fade-in       — Same as scroll-reveal (legacy class)
//   .story-section       — Same as scroll-reveal (legacy class)
//   .fade-in             — Fade in + slide up (0.7s, ease-out)
//   .reveal              — Fade in + slide up (0.7s, ease-out) (reference pages)
//   .narrative-card      — Fade in + slide up (0.8s, card style)
//   .synthesis-card      — Fade in + slide up (0.8s, card style)
//
// All elements get the .visible class added when they enter the viewport.
// The observer uses a 15% threshold for triggering.
//
// To add a new class: just add it to the SELECTORS array below.
// ============================================================================

(function() {
    'use strict';

    // All selectors that should animate on scroll
    var SELECTORS = [
        '.scroll-reveal',
        '.scroll-reveal-slow',
        '.story-fade-in',
        '.story-section',
        '.fade-in',
        '.reveal',
        '.narrative-card',
        '.synthesis-card',
        '.region-step-card'
    ].join(',');

    // Threshold: element is 15% visible
    var OPTIONS = {
        threshold: 0.15,
        rootMargin: '0px 0px -40px 0px'
    };

    function initObserver() {
        if (typeof IntersectionObserver === 'undefined') {
            // Fallback: make everything visible immediately
            document.querySelectorAll(SELECTORS).forEach(function(el) {
                el.classList.add('visible');
            });
            return;
        }

        var observer = new IntersectionObserver(function(entries) {
            entries.forEach(function(entry) {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    observer.unobserve(entry.target); // Once visible, stop watching
                }
            });
        }, OPTIONS);

        // Observe all matching elements
        document.querySelectorAll(SELECTORS).forEach(function(el) {
            // Skip elements that are already visible (e.g., above the fold)
            if (el.getBoundingClientRect().top < window.innerHeight * 0.85) {
                el.classList.add('visible');
            } else {
                observer.observe(el);
            }
        });
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initObserver);
    } else {
        initObserver();
    }
})();
