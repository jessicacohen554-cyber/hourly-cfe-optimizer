/**
 * Cinematic Scroll Background - Shared Module
 * Injects a fixed background image and creates a GSAP ScrollTrigger
 * animation that pans/zooms the image as the user scrolls.
 *
 * Requirements: GSAP + ScrollTrigger loaded before this script.
 * Include in <head>:
 *   <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
 *   <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/ScrollTrigger.min.js"></script>
 *   <link rel="stylesheet" href="styles/cinematic.css">
 * Include before </body>:
 *   <script src="js/cinematic-bg.js"></script>
 */
(function() {
    'use strict';

    // --- Inject background container ---
    var camera = document.createElement('div');
    camera.id = 'bgCamera';
    camera.setAttribute('aria-hidden', 'true');

    var img = document.createElement('img');
    img.id = 'bgImage';
    img.alt = '';
    // Resolve image path relative to the page (all pages are in dashboard/)
    img.src = (document.currentScript && document.currentScript.src)
        ? new URL('../134ECF38-1159-473B-809C-019548A1B759.png', document.currentScript.src).href
        : '134ECF38-1159-473B-809C-019548A1B759.png';

    camera.appendChild(img);
    document.body.insertBefore(camera, document.body.firstChild);

    // --- Reduced motion: static background ---
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        img.style.transform = 'translate(-15%, -15%) scale(1.0)';
        return;
    }

    // --- Wait for GSAP ---
    if (typeof gsap === 'undefined' || typeof ScrollTrigger === 'undefined') {
        console.warn('cinematic-bg.js: GSAP/ScrollTrigger not loaded. Skipping animation.');
        img.style.transform = 'translate(-15%, -15%) scale(1.0)';
        return;
    }

    gsap.registerPlugin(ScrollTrigger);

    // Initial position (centered with overscan for panning room)
    gsap.set(img, { xPercent: -15, yPercent: -15, scale: 1.0 });

    // --- Generate varied keyframes seeded by page path ---
    // Each page gets slightly different pan coordinates so the
    // background movement feels unique per page.
    var seed = 0;
    var path = window.location.pathname;
    for (var i = 0; i < path.length; i++) {
        seed = ((seed << 5) - seed + path.charCodeAt(i)) | 0;
    }
    function seededRand() {
        seed = (seed * 16807 + 0) % 2147483647;
        return (seed & 0x7fffffff) / 2147483647;
    }

    // Base keyframes - wide sweeping motions
    var baseKeyframes = [
        { x: -24, y: -8,  s: 1.08, d: 1.0 },
        { x: -5,  y: -22, s: 1.14, d: 1.0 },
        { x: -28, y: -12, s: 1.18, d: 1.2 },
        { x: -8,  y: -26, s: 1.12, d: 1.0 },
        { x: -22, y: -6,  s: 1.22, d: 1.2 },
        { x: -4,  y: -20, s: 1.15, d: 1.0 },
        { x: -15, y: -15, s: 1.05, d: 1.2 }
    ];

    // Add per-page variation (±4% position, ±0.03 scale)
    var keyframes = baseKeyframes.map(function(kf) {
        return {
            x: kf.x + (seededRand() - 0.5) * 8,
            y: kf.y + (seededRand() - 0.5) * 8,
            s: kf.s + (seededRand() - 0.5) * 0.06,
            d: kf.d
        };
    });
    // Last keyframe always returns near center
    keyframes[keyframes.length - 1] = { x: -15, y: -15, s: 1.05, d: 1.2 };

    // --- Build GSAP timeline ---
    var tl = gsap.timeline({
        scrollTrigger: {
            trigger: document.body,
            start: 'top top',
            end: 'bottom bottom',
            scrub: 1.8
        }
    });

    keyframes.forEach(function(kf) {
        tl.to(img, {
            xPercent: kf.x,
            yPercent: kf.y,
            scale: kf.s,
            ease: 'power2.inOut',
            duration: kf.d
        });
    });
})();
