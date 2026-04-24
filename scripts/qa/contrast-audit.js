#!/usr/bin/env node
/**
 * contrast-audit.js — WCAG 2.1 contrast audit for dashboard HTML pages.
 *
 * Parses each HTML file in dashboard/, resolves text+bg colors from linked CSS,
 * computes WCAG contrast ratios, and outputs JSON + Markdown reports.
 */

const fs = require('fs');
const path = require('path');
const cheerio = require('cheerio');
const csstree = require('css-tree');

// ── Helpers ──────────────────────────────────────────────────────────────────

function hexToRGB(hex) {
    hex = hex.replace('#', '');
    if (hex.length === 3) hex = hex[0]+hex[0]+hex[1]+hex[1]+hex[2]+hex[2];
    return {
        r: parseInt(hex.substring(0,2), 16),
        g: parseInt(hex.substring(2,4), 16),
        b: parseInt(hex.substring(4,6), 16),
        a: 1
    };
}

function rgbToHex(r, g, b) {
    const h = (v) => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, '0');
    return `#${h(r)}${h(g)}${h(b)}`.toUpperCase();
}

function parseColor(str) {
    if (!str) return null;
    str = str.trim().toLowerCase();
    if (str === 'transparent') return { r: 0, g: 0, b: 0, a: 0 };
    if (str === 'inherit' || str === 'initial' || str === 'unset' || str === 'currentcolor') return null;

    // Named colors (common ones)
    const named = {
        white: '#FFFFFF', black: '#000000', red: '#FF0000', blue: '#0000FF',
        green: '#008000', yellow: '#FFFF00', gray: '#808080', grey: '#808080',
        navy: '#000080', orange: '#FFA500', purple: '#800080', pink: '#FFC0CB',
        silver: '#C0C0C0', teal: '#008080', cyan: '#00FFFF', magenta: '#FF00FF',
        lime: '#00FF00', maroon: '#800000', olive: '#808000', aqua: '#00FFFF',
        indigo: '#4B0082', violet: '#EE82EE', coral: '#FF7F50', crimson: '#DC143C',
        darkblue: '#00008B', darkgreen: '#006400', darkgray: '#A9A9A9',
        darkgrey: '#A9A9A9', lightgray: '#D3D3D3', lightgrey: '#D3D3D3',
        whitesmoke: '#F5F5F5', lightblue: '#ADD8E6', steelblue: '#4682B4',
        slategray: '#708090', slategrey: '#708090',
    };
    if (named[str]) return hexToRGB(named[str]);

    // hex
    const hexMatch = str.match(/^#([0-9a-f]{3,8})$/);
    if (hexMatch) {
        const hex = hexMatch[1];
        if (hex.length === 3 || hex.length === 6) return hexToRGB(str);
        if (hex.length === 8) {
            const c = hexToRGB('#' + hex.substring(0,6));
            c.a = parseInt(hex.substring(6,8), 16) / 255;
            return c;
        }
        if (hex.length === 4) {
            const c = hexToRGB('#' + hex[0]+hex[0]+hex[1]+hex[1]+hex[2]+hex[2]);
            c.a = parseInt(hex[3]+hex[3], 16) / 255;
            return c;
        }
    }

    // rgb(a)
    const rgbaMatch = str.match(/rgba?\(\s*(\d+(?:\.\d+)?%?)\s*[,\s]\s*(\d+(?:\.\d+)?%?)\s*[,\s]\s*(\d+(?:\.\d+)?%?)\s*(?:[,/]\s*([\d.]+%?)\s*)?\)/);
    if (rgbaMatch) {
        const parseVal = (v) => v.endsWith('%') ? parseFloat(v) / 100 * 255 : parseFloat(v);
        return {
            r: parseVal(rgbaMatch[1]),
            g: parseVal(rgbaMatch[2]),
            b: parseVal(rgbaMatch[3]),
            a: rgbaMatch[4] ? (rgbaMatch[4].endsWith('%') ? parseFloat(rgbaMatch[4])/100 : parseFloat(rgbaMatch[4])) : 1
        };
    }

    // hsl(a)
    const hslaMatch = str.match(/hsla?\(\s*([\d.]+)\s*[,\s]\s*([\d.]+)%?\s*[,\s]\s*([\d.]+)%?\s*(?:[,/]\s*([\d.]+%?)\s*)?\)/);
    if (hslaMatch) {
        const h = parseFloat(hslaMatch[1]) / 360;
        const s = parseFloat(hslaMatch[2]) / 100;
        const l = parseFloat(hslaMatch[3]) / 100;
        const a = hslaMatch[4] ? (hslaMatch[4].endsWith('%') ? parseFloat(hslaMatch[4])/100 : parseFloat(hslaMatch[4])) : 1;
        // HSL to RGB
        let r, g, b;
        if (s === 0) { r = g = b = l; }
        else {
            const hue2rgb = (p, q, t) => {
                if (t < 0) t += 1; if (t > 1) t -= 1;
                if (t < 1/6) return p + (q-p)*6*t;
                if (t < 1/2) return q;
                if (t < 2/3) return p + (q-p)*(2/3-t)*6;
                return p;
            };
            const q = l < 0.5 ? l*(1+s) : l+s-l*s;
            const p = 2*l - q;
            r = hue2rgb(p, q, h+1/3);
            g = hue2rgb(p, q, h);
            b = hue2rgb(p, q, h-1/3);
        }
        return { r: r*255, g: g*255, b: b*255, a };
    }

    return null;
}

function alphaComposite(fg, bg) {
    if (!fg || fg.a === 0) return bg;
    if (!bg) return fg;
    const a = fg.a + bg.a * (1 - fg.a);
    if (a === 0) return { r: 0, g: 0, b: 0, a: 0 };
    return {
        r: (fg.r * fg.a + bg.r * bg.a * (1 - fg.a)) / a,
        g: (fg.g * fg.a + bg.g * bg.a * (1 - fg.a)) / a,
        b: (fg.b * fg.a + bg.b * bg.a * (1 - fg.a)) / a,
        a
    };
}

function relativeLuminance(r, g, b) {
    const srgb = [r, g, b].map(v => {
        v = v / 255;
        return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * srgb[0] + 0.7152 * srgb[1] + 0.0722 * srgb[2];
}

function contrastRatio(c1, c2) {
    const l1 = relativeLuminance(c1.r, c1.g, c1.b);
    const l2 = relativeLuminance(c2.r, c2.g, c2.b);
    const lighter = Math.max(l1, l2);
    const darker = Math.min(l1, l2);
    return (lighter + 0.05) / (darker + 0.05);
}

function suggestColor(bgColor, requiredRatio, currentTextColor) {
    // Try lightening/darkening the current text color to meet ratio
    const bgLum = relativeLuminance(bgColor.r, bgColor.g, bgColor.b);
    // Try lighter colors first (for dark bg)
    for (let step = 0; step <= 255; step += 5) {
        const r = Math.min(255, currentTextColor.r + step);
        const g = Math.min(255, currentTextColor.g + step);
        const b = Math.min(255, currentTextColor.b + step);
        const lum = relativeLuminance(r, g, b);
        const ratio = (Math.max(lum, bgLum) + 0.05) / (Math.min(lum, bgLum) + 0.05);
        if (ratio >= requiredRatio) return rgbToHex(r, g, b);
    }
    // Try darker
    for (let step = 0; step <= 255; step += 5) {
        const r = Math.max(0, currentTextColor.r - step);
        const g = Math.max(0, currentTextColor.g - step);
        const b = Math.max(0, currentTextColor.b - step);
        const lum = relativeLuminance(r, g, b);
        const ratio = (Math.max(lum, bgLum) + 0.05) / (Math.min(lum, bgLum) + 0.05);
        if (ratio >= requiredRatio) return rgbToHex(r, g, b);
    }
    return bgLum > 0.5 ? '#1E293B' : '#E2E8F0';
}


// ── CSS Parsing & Variable Resolution ────────────────────────────────────────

function loadCSS(htmlFile, $) {
    const dir = path.dirname(htmlFile);
    const cssFiles = [];
    $('link[rel="stylesheet"]').each((_, el) => {
        const href = $(el).attr('href');
        if (href && !href.startsWith('http')) {
            const cssPath = path.resolve(dir, href);
            if (fs.existsSync(cssPath)) cssFiles.push(cssPath);
        }
    });
    // Also check for <style> blocks
    let allCSS = '';
    cssFiles.forEach(f => {
        allCSS += fs.readFileSync(f, 'utf8') + '\n';
    });
    $('style').each((_, el) => {
        allCSS += $(el).html() + '\n';
    });
    return { allCSS, cssFiles };
}

function extractCSSVariables(cssText) {
    const vars = {};
    // Match variable declarations: --name: value;
    const regex = /(-{2}[\w-]+)\s*:\s*([^;]+)/g;
    let m;
    while ((m = regex.exec(cssText)) !== null) {
        vars[m[1]] = m[2].trim();
    }
    return vars;
}

function resolveVar(value, vars, depth = 0) {
    if (!value || depth > 10) return value;
    // Resolve var(--name, fallback)
    return value.replace(/var\(\s*(-{2}[\w-]+)\s*(?:,\s*([^)]+))?\s*\)/g, (match, name, fallback) => {
        let resolved = vars[name];
        if (resolved) {
            return resolveVar(resolved, vars, depth + 1);
        }
        if (fallback) {
            return resolveVar(fallback.trim(), vars, depth + 1);
        }
        return match;
    });
}

// ── CSS Rule Matching ────────────────────────────────────────────────────────

function buildSelectorIndex(cssText) {
    const rules = [];
    try {
        const ast = csstree.parse(cssText, { parseRulePrelude: false, parseCustomProperty: false });
        csstree.walk(ast, {
            visit: 'Rule',
            enter(node) {
                const selectorText = csstree.generate(node.prelude);
                const declarations = {};
                let hasImportant = {};
                if (node.block) {
                    csstree.walk(node.block, {
                        visit: 'Declaration',
                        enter(decl) {
                            const prop = decl.property;
                            const val = csstree.generate(decl.value);
                            declarations[prop] = val;
                            if (decl.important) hasImportant[prop] = true;
                        }
                    });
                }
                // Split compound selectors
                selectorText.split(',').forEach(sel => {
                    sel = sel.trim();
                    if (sel) {
                        rules.push({
                            selector: sel,
                            declarations,
                            important: hasImportant,
                            specificity: calcSpecificity(sel)
                        });
                    }
                });
            }
        });
    } catch (e) {
        // fallback: regex-based extraction
    }
    return rules;
}

function calcSpecificity(selector) {
    // Simple specificity calculation: [inline, ids, classes, elements]
    let ids = 0, classes = 0, elements = 0;
    // Remove :not() content but count its internals
    const cleaned = selector.replace(/:not\(([^)]*)\)/g, (_, inner) => {
        const s = calcSpecificity(inner);
        ids += s[1]; classes += s[2]; elements += s[3];
        return '';
    });
    // Count IDs
    ids += (cleaned.match(/#[\w-]+/g) || []).length;
    // Count classes, pseudo-classes, attributes
    classes += (cleaned.match(/\.[\w-]+/g) || []).length;
    classes += (cleaned.match(/\[[\w-]+/g) || []).length;
    classes += (cleaned.match(/:[\w-]+/g) || []).filter(p =>
        !p.match(/^:(before|after|first-line|first-letter|selection|placeholder|backdrop)/)
    ).length;
    // Count elements and pseudo-elements
    const withoutIds = cleaned.replace(/#[\w-]+/g, '');
    const withoutClasses = withoutIds.replace(/\.[\w-]+/g, '').replace(/\[[\w-]+[^\]]*\]/g, '').replace(/:[\w-]+/g, '');
    elements += (withoutClasses.match(/(?:^|[\s>+~])[\w]+/g) || []).length;
    elements += (cleaned.match(/::[\w-]+/g) || []).length;

    return [0, ids, classes, elements];
}

function compareSpecificity(a, b) {
    for (let i = 0; i < 4; i++) {
        if (a[i] !== b[i]) return a[i] - b[i];
    }
    return 0;
}

function matchesSelector($, el, selector) {
    try {
        // Use cheerio's .is() for matching
        return $(el).is(selector);
    } catch (e) {
        return false;
    }
}

function getComputedProperty($, el, property, cssRules, cssVars) {
    // 1. Check inline style
    const style = $(el).attr('style');
    if (style) {
        const escapedProp = property.replace(/-/g, '\\-');
        const inlineMatch = style.match(new RegExp(`(?:^|;)\\s*${escapedProp}\\s*:\\s*([^;!]+)(!important)?`, 'i'));
        if (inlineMatch) {
            const val = resolveVar(inlineMatch[1].trim(), cssVars);
            return { value: val, important: !!inlineMatch[2], specificity: [1, 0, 0, 0] };
        }
    }

    // 2. Find matching CSS rules
    let best = null;
    let bestIndex = -1;
    for (let i = 0; i < cssRules.length; i++) {
        const rule = cssRules[i];
        if (!(property in rule.declarations)) continue;
        try {
            if (!matchesSelector($, el, rule.selector)) continue;
        } catch (e) {
            continue;
        }

        const isImportant = rule.important[property] || false;
        const spec = rule.specificity;

        if (!best) {
            best = { value: rule.declarations[property], important: isImportant, specificity: spec };
            bestIndex = i;
        } else {
            // !important beats non-important
            if (isImportant && !best.important) {
                best = { value: rule.declarations[property], important: isImportant, specificity: spec };
                bestIndex = i;
            } else if (isImportant === best.important) {
                // Higher specificity wins; later rule wins at equal specificity
                if (compareSpecificity(spec, best.specificity) >= 0) {
                    best = { value: rule.declarations[property], important: isImportant, specificity: spec };
                    bestIndex = i;
                }
            }
        }
    }

    // Inline with !important trumps everything except inline, which we handled above
    // Inline without !important trumps CSS without !important
    if (style && !best?.important) {
        const inlineMatch = style.match(new RegExp(`(?:^|;)\\s*${property.replace('-', '\\-')}\\s*:\\s*([^;]+)`, 'i'));
        if (inlineMatch) {
            const val = resolveVar(inlineMatch[1].trim(), cssVars);
            return { value: val, important: false, specificity: [1, 0, 0, 0] };
        }
    }

    if (best) {
        best.value = resolveVar(best.value, cssVars);
        return best;
    }
    return null;
}

// ── Element Text Color Resolution ────────────────────────────────────────────

function resolveTextColor($, el, cssRules, cssVars) {
    // Walk up from element to find computed color
    let current = el;
    while (current && $(current).length) {
        const result = getComputedProperty($, current, 'color', cssRules, cssVars);
        if (result) {
            const c = parseColor(result.value);
            if (c) return { color: c, raw: result.value };
        }
        current = $(current).parent()[0];
        if (!current || $(current).is('html') || $(current).is('[cheerio-root]')) break;
    }
    // Default
    return { color: { r: 0, g: 0, b: 0, a: 1 }, raw: '#000000' };
}

function resolveOpacity($, el, cssRules, cssVars) {
    let opacity = 1;
    let current = el;
    while (current && $(current).length) {
        const result = getComputedProperty($, current, 'opacity', cssRules, cssVars);
        if (result) {
            const val = parseFloat(result.value);
            if (!isNaN(val)) opacity *= val;
        }
        current = $(current).parent()[0];
        if (!current || $(current).is('html') || $(current).is('[cheerio-root]')) break;
    }
    return opacity;
}

// ── Background Color Resolution ──────────────────────────────────────────────

function resolveEffectiveBg($, el, cssRules, cssVars, isDarkPage) {
    // Walk up from element, compositing semi-transparent backgrounds
    const bgStack = [];
    let current = el;
    let hasBackdropFilter = false;

    while (current && $(current).length) {
        // Check for backdrop-filter
        const bf = getComputedProperty($, current, 'backdrop-filter', cssRules, cssVars);
        if (bf && bf.value && bf.value.includes('blur')) hasBackdropFilter = true;

        // Check background-color first, then background shorthand
        let bgResult = getComputedProperty($, current, 'background-color', cssRules, cssVars);
        let bgResult2 = getComputedProperty($, current, 'background', cssRules, cssVars);

        // Use whichever has higher specificity, preferring background-color if equal
        let chosen = bgResult;
        if (bgResult2 && (!bgResult ||
            (bgResult2.important && !bgResult.important) ||
            (bgResult2.important === bgResult.important && compareSpecificity(bgResult2.specificity, bgResult.specificity) > 0))) {
            chosen = bgResult2;
        }

        if (chosen && chosen.value) {
            let bgVal = chosen.value;
            // Handle gradients - extract dominant color (average of first and last for linear)
            if (bgVal.includes('gradient')) {
                const colorMatches = bgVal.match(/(#[0-9a-fA-F]{3,8}|rgba?\([^)]+\)|hsla?\([^)]+\))/g);
                if (colorMatches && colorMatches.length > 0) {
                    // Use first color of gradient (darkest for dark gradients)
                    bgVal = colorMatches[0];
                }
            }
            // Strip url(), no-repeat, etc.
            bgVal = bgVal.replace(/url\([^)]*\)/g, '').replace(/no-repeat|repeat|center|cover|contain|left|right|top|bottom|\/[\d%\s]+/gi, '').trim();
            // Strip 'none'
            if (bgVal === 'none' || bgVal === '') bgVal = null;

            if (bgVal) {
                const c = parseColor(bgVal);
                if (c) {
                    bgStack.push(c);
                    if (c.a >= 0.95) break; // Opaque enough
                }
            }
        }

        current = $(current).parent()[0];
        if (!current || $(current).is('[cheerio-root]')) break;
    }

    // Composite from bottom (page bg) to top
    let baseBg = isDarkPage ? hexToRGB('#0B1120') : hexToRGB('#FFFFFF');
    bgStack.reverse();
    let effective = baseBg;
    for (const bg of bgStack) {
        effective = alphaComposite(bg, effective);
    }

    return { color: effective, raw: rgbToHex(effective.r, effective.g, effective.b), hasBackdropFilter };
}

// ── Font Size Detection ──────────────────────────────────────────────────────

function resolveFontSize($, el, cssRules, cssVars) {
    let current = el;
    while (current && $(current).length) {
        const result = getComputedProperty($, current, 'font-size', cssRules, cssVars);
        if (result) {
            const val = result.value;
            // Parse px
            const pxMatch = val.match(/([\d.]+)\s*px/);
            if (pxMatch) return parseFloat(pxMatch[1]);
            // Parse rem (assume 16px base)
            const remMatch = val.match(/([\d.]+)\s*rem/);
            if (remMatch) return parseFloat(remMatch[1]) * 16;
            // Parse em (approximate)
            const emMatch = val.match(/([\d.]+)\s*em/);
            if (emMatch) return parseFloat(emMatch[1]) * 16;
        }
        current = $(current).parent()[0];
        if (!current || $(current).is('html') || $(current).is('[cheerio-root]')) break;
    }
    return 16; // default
}

function resolveFontWeight($, el, cssRules, cssVars) {
    let current = el;
    while (current && $(current).length) {
        const result = getComputedProperty($, current, 'font-weight', cssRules, cssVars);
        if (result) {
            const val = result.value;
            if (val === 'bold') return 700;
            if (val === 'normal') return 400;
            const n = parseInt(val);
            if (!isNaN(n)) return n;
        }
        // Check tag
        const tag = $(current).prop('tagName')?.toLowerCase();
        if (tag === 'strong' || tag === 'b' || tag === 'th') return 700;
        if (tag && tag.match(/^h[1-6]$/)) return 700;

        current = $(current).parent()[0];
        if (!current || $(current).is('html') || $(current).is('[cheerio-root]')) break;
    }
    return 400;
}

// ── Selector Path Generation ─────────────────────────────────────────────────

function getSelectorPath($, el) {
    const parts = [];
    let current = el;
    let depth = 0;
    while (current && $(current).length && depth < 5) {
        const tag = $(current).prop('tagName')?.toLowerCase();
        if (!tag || tag === '[document]') break;
        let part = tag;
        const cls = $(current).attr('class');
        if (cls) {
            part += '.' + cls.split(/\s+/).filter(c => c).slice(0, 2).join('.');
        }
        const id = $(current).attr('id');
        if (id) part = tag + '#' + id;
        parts.unshift(part);
        current = $(current).parent()[0];
        depth++;
    }
    return parts.join(' > ');
}

// ── Dark Page Detection ──────────────────────────────────────────────────────

function isDarkThemePage($, cssText, cssVars) {
    // Check body/html background
    const body = $('body');
    const bodyClass = body.attr('class') || '';
    const bodyStyle = body.attr('style') || '';

    // Check explicit dark indicators
    if (bodyClass.match(/dark|navy|night/i)) return true;
    if (bodyStyle.match(/background.*#[01][0-9a-fA-F]/i)) return true;

    // Check --bg-page variable
    const bgPage = cssVars['--bg-page'];
    if (bgPage) {
        const c = parseColor(resolveVar(bgPage, cssVars));
        if (c) {
            const lum = relativeLuminance(c.r, c.g, c.b);
            return lum < 0.2;
        }
    }

    // Check body background in CSS
    const bodyBgMatch = cssText.match(/body\s*\{[^}]*background(?:-color)?\s*:\s*([^;]+)/);
    if (bodyBgMatch) {
        const resolved = resolveVar(bodyBgMatch[1].trim(), cssVars);
        const c = parseColor(resolved);
        if (c) return relativeLuminance(c.r, c.g, c.b) < 0.2;
    }

    return false; // default to light
}

// ── Main Audit Logic ─────────────────────────────────────────────────────────

function findHTMLFiles(dir) {
    const files = [];
    function walk(d) {
        for (const entry of fs.readdirSync(d, { withFileTypes: true })) {
            const full = path.join(d, entry.name);
            if (entry.isDirectory()) walk(full);
            else if (entry.name.endsWith('.html')) files.push(full);
        }
    }
    walk(dir);
    return files;
}

const TEXT_SELECTORS = 'h1, h2, h3, h4, h5, h6, p, span, li, a, td, th, label, blockquote, figcaption, button, small, em, strong, b, dt, dd, summary, caption';
const TEXT_CLASS_PATTERNS = /text|title|subtitle|label|description|caption|tag|stat-value|stat-label|finding-label|finding-desc|badge|callout|insight|heading|val|lbl/i;

function isTextElement($, el) {
    const tag = $(el).prop('tagName')?.toLowerCase();
    if (!tag) return false;

    // Direct text element tags
    if (tag.match(/^(h[1-6]|p|span|li|a|td|th|label|blockquote|figcaption|button|small|em|strong|b|dt|dd|summary|caption)$/)) {
        return true;
    }

    // Check class patterns
    const cls = $(el).attr('class') || '';
    if (TEXT_CLASS_PATTERNS.test(cls)) return true;

    // div with direct text content and text-like class
    if (tag === 'div' && cls.match(TEXT_CLASS_PATTERNS)) return true;

    return false;
}

function getDirectText($, el) {
    // Get text that's directly in this element (not in children elements)
    let text = '';
    $(el).contents().each((_, child) => {
        if (child.type === 'text') text += child.data;
    });
    // Also include full text if this is a leaf-like element
    const children = $(el).children();
    if (children.length === 0 || (children.length <= 2 && $(el).text().length < 200)) {
        text = $(el).text();
    }
    return text.trim();
}

function auditFile(htmlPath) {
    const html = fs.readFileSync(htmlPath, 'utf8');
    const $ = cheerio.load(html);
    const relPath = path.relative(process.cwd(), htmlPath);

    // Load all CSS
    const { allCSS } = loadCSS(htmlPath, $);
    const cssVars = extractCSSVariables(allCSS);
    const cssRules = buildSelectorIndex(allCSS);
    const darkPage = isDarkThemePage($, allCSS, cssVars);

    const results = { passes: [], failures: [] };

    // Find all text elements
    const elements = new Set();
    $(TEXT_SELECTORS).each((_, el) => elements.add(el));
    $('*').each((_, el) => {
        const cls = $(el).attr('class') || '';
        if (TEXT_CLASS_PATTERNS.test(cls)) elements.add(el);
    });

    for (const el of elements) {
        const text = getDirectText($, el);
        if (!text || text.length < 2) continue;

        // Skip hidden elements
        const display = getComputedProperty($, el, 'display', cssRules, cssVars);
        if (display && display.value === 'none') continue;
        const visibility = getComputedProperty($, el, 'visibility', cssRules, cssVars);
        if (visibility && visibility.value === 'hidden') continue;

        // Determine if this element is in a dark context (panel-navy, section-dark, etc.)
        let isDarkContext = darkPage;
        let current = el;
        while (current && $(current).length) {
            const cls = $(current).attr('class') || '';
            if (cls.match(/panel-navy|section-dark|dark-section|navy-section|bg-dark/)) {
                isDarkContext = true;
                break;
            }
            if (cls.match(/section-light|bg-light|bg-white/)) {
                isDarkContext = false;
                break;
            }
            current = $(current).parent()[0];
            if (!current || $(current).is('[cheerio-root]')) break;
        }

        const textColorResult = resolveTextColor($, el, cssRules, cssVars);
        let textColor = textColorResult.color;
        const bgResult = resolveEffectiveBg($, el, cssRules, cssVars, isDarkContext);
        let bgColor = bgResult.color;

        // Apply opacity — but skip animation-initial states.
        // Elements starting at opacity:0 with a transition on opacity are
        // animated to opacity:1 by JavaScript. Audit their visible (final) state.
        let opacity = resolveOpacity($, el, cssRules, cssVars);
        if (opacity < 0.1) {
            // Check if any ancestor has opacity:0 + transition containing 'opacity'
            // OR has a class indicating animation (fade-in, narrative-card, etc.)
            let isAnimated = false;
            let anc = el;
            while (anc && $(anc).length) {
                const cls = $(anc).attr('class') || '';
                // Check class-based animation patterns
                if (cls.match(/\bfade-in\b|\bstory-fade-in\b|\banimate-in\b|\bslide-in\b|\bnarrative-card\b/)) {
                    isAnimated = true;
                    break;
                }
                // Check CSS: if element has opacity < 0.1 AND transition includes 'opacity'
                const opProp = getComputedProperty($, anc, 'opacity', cssRules, cssVars);
                if (opProp && parseFloat(opProp.value) < 0.1) {
                    const transProp = getComputedProperty($, anc, 'transition', cssRules, cssVars);
                    if (transProp && transProp.value && transProp.value.includes('opacity')) {
                        isAnimated = true;
                        break;
                    }
                    // If an element has explicit opacity:0 in CSS (not inherited default),
                    // it's almost certainly an animation initial state waiting for JS observer.
                    // Treat as animated (visible) for contrast auditing.
                    isAnimated = true;
                    break;
                }
                anc = $(anc).parent()[0];
                if (!anc || $(anc).is('[cheerio-root]')) break;
            }
            if (isAnimated) opacity = 1; // Treat as visible
        }
        if (opacity < 1) {
            textColor = { ...textColor, a: textColor.a * opacity };
        }

        // Composite text color with alpha onto background
        const effectiveText = textColor.a < 1 ? alphaComposite(textColor, bgColor) : textColor;

        const ratio = contrastRatio(effectiveText, bgColor);
        const fontSize = resolveFontSize($, el, cssRules, cssVars);
        const fontWeight = resolveFontWeight($, el, cssRules, cssVars);
        const isLargeText = fontSize >= 18 || (fontSize >= 14 && fontWeight >= 700);
        const requiredRatio = isLargeText ? 3.0 : 4.5;

        const selectorPath = getSelectorPath($, el);
        const textSnippet = text.substring(0, 80);
        const textHex = rgbToHex(effectiveText.r, effectiveText.g, effectiveText.b);
        const bgHex = rgbToHex(bgColor.r, bgColor.g, bgColor.b);

        const entry = {
            file: relPath,
            selector_path: selectorPath,
            text_snippet: textSnippet,
            text_color: textHex,
            text_color_raw: textColorResult.raw,
            effective_bg: bgHex,
            contrast_ratio: Math.round(ratio * 100) / 100,
            required_ratio: requiredRatio,
            font_size: `${Math.round(fontSize)}px`,
            font_weight: fontWeight,
            is_large_text: isLargeText,
            has_backdrop_filter: bgResult.hasBackdropFilter,
            dark_context: isDarkContext
        };

        if (ratio >= requiredRatio) {
            results.passes.push(entry);
        } else {
            // Classify severity
            const gap = requiredRatio - ratio;
            let severity;
            if (ratio < 2.0) severity = 'critical';
            else if (ratio < 3.0) severity = 'critical';
            else if (ratio < requiredRatio * 0.8) severity = 'major';
            else severity = 'minor';

            entry.severity = severity;
            entry.suggestion = `Change text to ${suggestColor(bgColor, requiredRatio, effectiveText)} or adjust for ${requiredRatio}:1 on bg ${bgHex}`;
            results.failures.push(entry);
        }
    }

    return results;
}

// ── Report Generation ────────────────────────────────────────────────────────

function generateReport(allResults) {
    const summary = {
        total_elements: 0,
        passing: 0,
        failing: 0,
        critical: 0,
        major: 0,
        minor: 0
    };

    const allFailures = [];
    const allPasses = [];

    for (const result of allResults) {
        summary.total_elements += result.passes.length + result.failures.length;
        summary.passing += result.passes.length;
        summary.failing += result.failures.length;
        for (const f of result.failures) {
            if (f.severity === 'critical') summary.critical++;
            else if (f.severity === 'major') summary.major++;
            else summary.minor++;
            allFailures.push(f);
        }
        allPasses.push(...result.passes);
    }

    // Sort failures: critical first, then major, then minor
    const severityOrder = { critical: 0, major: 1, minor: 2 };
    allFailures.sort((a, b) => severityOrder[a.severity] - severityOrder[b.severity]);

    const report = {
        generated: new Date().toISOString(),
        summary,
        failures: allFailures,
        passes: allPasses
    };

    return report;
}

function generateMarkdown(report) {
    let md = `# Contrast Audit Report\n\n`;
    md += `Generated: ${report.generated}\n\n`;
    md += `## Summary\n\n`;
    md += `| Metric | Count |\n|--------|-------|\n`;
    md += `| Total elements | ${report.summary.total_elements} |\n`;
    md += `| Passing | ${report.summary.passing} |\n`;
    md += `| **Failing** | **${report.summary.failing}** |\n`;
    md += `| Critical | ${report.summary.critical} |\n`;
    md += `| Major | ${report.summary.major} |\n`;
    md += `| Minor | ${report.summary.minor} |\n\n`;

    // Group by file
    const byFile = {};
    for (const f of report.failures) {
        if (!byFile[f.file]) byFile[f.file] = [];
        byFile[f.file].push(f);
    }

    for (const [file, failures] of Object.entries(byFile)) {
        md += `## ${file}\n\n`;
        md += `**${failures.length} failures**\n\n`;
        for (const f of failures) {
            md += `### [${f.severity.toUpperCase()}] ${f.selector_path}\n\n`;
            md += `- **Text**: "${f.text_snippet}"\n`;
            md += `- **Color**: ${f.text_color} on ${f.effective_bg}\n`;
            md += `- **Ratio**: ${f.contrast_ratio}:1 (need ${f.required_ratio}:1)\n`;
            md += `- **Font**: ${f.font_size}, weight ${f.font_weight}\n`;
            md += `- **Fix**: ${f.suggestion}\n\n`;
        }
    }

    return md;
}

// ── Main ─────────────────────────────────────────────────────────────────────

console.log('Contrast Audit — WCAG 2.1 AA\n');

const dashboardDir = path.resolve(__dirname, 'dashboard');
const htmlFiles = findHTMLFiles(dashboardDir);
console.log(`Found ${htmlFiles.length} HTML files in dashboard/\n`);

const allResults = [];
for (const file of htmlFiles) {
    const relPath = path.relative(process.cwd(), file);
    process.stdout.write(`Auditing ${relPath}...`);
    try {
        const result = auditFile(file);
        allResults.push(result);
        const fails = result.failures.length;
        console.log(` ${result.passes.length} pass, ${fails} fail${fails > 0 ? ' ⚠' : ''}`);
    } catch (e) {
        console.log(` ERROR: ${e.message}`);
    }
}

const report = generateReport(allResults);

// Write JSON report
fs.writeFileSync('contrast-audit-report.json', JSON.stringify(report, null, 2));
console.log(`\nJSON report: contrast-audit-report.json`);

// Write Markdown report
const md = generateMarkdown(report);
fs.writeFileSync('contrast-audit-report.md', md);
console.log(`Markdown report: contrast-audit-report.md`);

// Print summary
console.log(`\n════════════════════════════════════════`);
console.log(`  SUMMARY`);
console.log(`════════════════════════════════════════`);
console.log(`  Total elements audited: ${report.summary.total_elements}`);
console.log(`  Passing:  ${report.summary.passing}`);
console.log(`  Failing:  ${report.summary.failing}`);
console.log(`    Critical: ${report.summary.critical}`);
console.log(`    Major:    ${report.summary.major}`);
console.log(`    Minor:    ${report.summary.minor}`);
console.log(`════════════════════════════════════════\n`);

// Calibration check
const calibrationCards = report.failures.filter(f =>
    f.file.includes('gen_policy_conditions') &&
    f.selector_path.includes('stat-label')
);
if (calibrationCards.length >= 4) {
    console.log('✓ CALIBRATION PASSED: Carbon pricing threshold cards detected as failures');
} else {
    console.log('✗ CALIBRATION FAILED: Expected 4 stat-label failures in gen_policy_conditions.html');
    console.log(`  Found ${calibrationCards.length} matching failures`);
    // Show what we found for debugging
    const allPolicyFailures = report.failures.filter(f => f.file.includes('gen_policy_conditions'));
    console.log(`  Total failures in gen_policy_conditions.html: ${allPolicyFailures.length}`);
    if (allPolicyFailures.length > 0) {
        console.log('  Sample failures:');
        allPolicyFailures.slice(0, 5).forEach(f => {
            console.log(`    ${f.selector_path}: "${f.text_snippet}" - ${f.text_color} on ${f.effective_bg} (${f.contrast_ratio}:1)`);
        });
    }
}
