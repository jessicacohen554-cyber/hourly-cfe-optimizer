// ============================================================================
// CHART COLORS - Canonical color constants for Chart.js
// ============================================================================
// Single source of truth for all chart colors. NEVER hardcode hex values
// in Chart.js dataset configurations - always use these constants.
//
// Usage: <script src="js/chart-colors.js"></script>
//        Then reference: RESOURCE_COLORS.solar, ISO_COLORS.CAISO, etc.
// ============================================================================

var RESOURCE_COLORS = {
    // Full opacity (borders, lines)
    solar:        '#F59E0B',
    wind:         '#22C55E',
    offshoreWind: '#009688',
    hydro:        '#0EA5E9',
    nuclear:      '#6366F1',
    ccs:          '#64748B',
    cleanFirm:    '#6366F1',
    battery:      '#06B6D4',   // Battery 4hr - cyan
    battery4:     '#06B6D4',   // Battery 4hr alias
    battery8:     '#0891B2',   // Battery 8hr - deep cyan
    ldes:         '#E91E63',
    greenH2:      '#10B981',
    geothermal:   '#D97706',
    storage:      '#EF4444',
    gap:          '#D1D5DB',
    fossilGas:    '#6B7280',
    fossilCoal:   '#374151',
    fossilOil:    '#92400E',

    // Transparent fills (55% opacity)
    solarT:        'rgba(245, 158, 11, 0.55)',
    windT:         'rgba(34, 197, 94, 0.55)',
    offshoreWindT: 'rgba(0, 150, 136, 0.55)',
    hydroT:        'rgba(14, 165, 233, 0.55)',
    nuclearT:      'rgba(99, 102, 241, 0.55)',
    ccsT:          'rgba(100, 116, 139, 0.55)',
    cleanFirmT:    'rgba(99, 102, 241, 0.55)',
    batteryT:      'rgba(6, 182, 212, 0.55)',   // Battery 4hr
    battery4T:     'rgba(6, 182, 212, 0.55)',   // Battery 4hr alias
    battery8T:     'rgba(8, 145, 178, 0.55)',    // Battery 8hr
    ldesT:         'rgba(233, 30, 99, 0.55)',
    greenH2T:      'rgba(16, 185, 129, 0.55)',
    geothermalT:   'rgba(217, 119, 6, 0.55)',
    storageT:      'rgba(239, 68, 68, 0.55)',
    gapT:          'rgba(209, 213, 219, 0.55)',

    // Light backgrounds (8% opacity)
    solarBg:        'rgba(245, 158, 11, 0.08)',
    windBg:         'rgba(34, 197, 94, 0.08)',
    offshoreWindBg: 'rgba(0, 150, 136, 0.08)',
    hydroBg:        'rgba(14, 165, 233, 0.08)',
    nuclearBg:      'rgba(99, 102, 241, 0.08)',
    ccsBg:          'rgba(100, 116, 139, 0.08)',
    batteryBg:      'rgba(6, 182, 212, 0.08)',   // Battery 4hr
    battery4Bg:     'rgba(6, 182, 212, 0.08)',
    battery8Bg:     'rgba(8, 145, 178, 0.08)',
    ldesBg:         'rgba(233, 30, 99, 0.08)',
    storageBg:      'rgba(239, 68, 68, 0.08)'
};

var ISO_COLORS = {
    // Full opacity (borders, lines, active buttons)
    CAISO: '#F59E0B',
    ERCOT: '#22C55E',
    PJM:   '#0EA5E9',
    NYISO: '#E91E63',
    NEISO: '#9C27B0',
    MISO:  '#F97316',
    SPP:   '#14B8A6',

    // Transparent fills (12% opacity)
    CAISO_T: 'rgba(245, 158, 11, 0.12)',
    ERCOT_T: 'rgba(34, 197, 94, 0.12)',
    PJM_T:   'rgba(14, 165, 233, 0.12)',
    NYISO_T: 'rgba(233, 30, 99, 0.12)',
    NEISO_T: 'rgba(156, 39, 176, 0.12)',
    MISO_T:  'rgba(249, 115, 22, 0.12)',
    SPP_T:   'rgba(20, 184, 166, 0.12)',

    // Ordered list (for iteration)
    list: ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP'],
    colors: ['#F59E0B', '#22C55E', '#0EA5E9', '#E91E63', '#9C27B0', '#F97316', '#14B8A6'],
    fills:  [
        'rgba(245, 158, 11, 0.12)', 'rgba(34, 197, 94, 0.12)',
        'rgba(14, 165, 233, 0.12)', 'rgba(233, 30, 99, 0.12)',
        'rgba(156, 39, 176, 0.12)', 'rgba(249, 115, 22, 0.12)',
        'rgba(20, 184, 166, 0.12)'
    ]
};

// Semantic colors for positive/negative/warning values
var SEMANTIC_COLORS = {
    positive: '#16A34A',
    negative: '#DC2626',
    warning:  '#D97706',
    info:     '#0284C7',
    accent:   '#38bdf8',
    muted:    '#6B7280'
};

// ============================================================================
// withAlpha() - Create semi-transparent fill from any hex color
// ============================================================================
// Usage: withAlpha(RESOURCE_COLORS.solar, 0.15) → 'rgba(245,158,11,0.15)'
// Pattern: borderColor: RESOURCE_COLORS.solar, backgroundColor: withAlpha(RESOURCE_COLORS.solar, 0.15)
function withAlpha(hex, alpha) {
    if (!hex || hex.charAt(0) !== '#') return hex;
    var r = parseInt(hex.slice(1, 3), 16);
    var g = parseInt(hex.slice(3, 5), 16);
    var b = parseInt(hex.slice(5, 7), 16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
}

// ============================================================================
// OUTLINE / SOLID DATASET FACTORIES
// ============================================================================
// outlineDataset: semi-transparent fill + saturated border (default for new-build)
// solidDataset:   saturated fill + matching border (for existing clean energy)
//
// Usage:
//   datasets: [
//     outlineDataset(RESOURCE_COLORS.solar, 'New Solar', [10, 20, 30]),
//     solidDataset(RESOURCE_COLORS.nuclear, 'Existing Nuclear', [50, 50, 50])
//   ]
// ============================================================================

function outlineDataset(color, label, data, opts) {
    opts = opts || {};
    var result = {
        label: label,
        data: data,
        backgroundColor: withAlpha(color, opts.fillAlpha || 0.15),
        borderColor: color,
        borderWidth: opts.borderWidth || 2,
        borderRadius: opts.borderRadius || 3,
        hoverBackgroundColor: withAlpha(color, 0.35),
        hoverBorderColor: color,
        hoverBorderWidth: (opts.borderWidth || 2) + 0.5
    };
    if (opts.extra) {
        for (var k in opts.extra) { result[k] = opts.extra[k]; }
    }
    return result;
}

function solidDataset(color, label, data, opts) {
    opts = opts || {};
    var result = {
        label: label,
        data: data,
        backgroundColor: withAlpha(color, opts.fillAlpha || 0.85),
        borderColor: color,
        borderWidth: opts.borderWidth || 2,
        borderRadius: opts.borderRadius || 3
    };
    if (opts.extra) {
        for (var k in opts.extra) { result[k] = opts.extra[k]; }
    }
    return result;
}

// Resource generation stack order (bottom to top for stacked charts)
var RESOURCE_STACK_ORDER = ['nuclear', 'geothermal', 'ccs', 'hydro', 'offshoreWind', 'wind', 'solar', 'battery', 'battery8', 'ldes', 'greenH2'];

// Curtailment stack order (bottom to top - reverse of generation: solar curtailed first, nuclear last)
var CURTAILMENT_STACK_ORDER = ['solar', 'wind', 'offshoreWind', 'hydro', 'ccs', 'geothermal', 'nuclear'];

// ============================================================================
// CHART.JS DEFAULT FONT SIZES - Desktop readability
// Applied after DOM loads to ensure Chart.js is available
// ============================================================================
function _applyChartDefaults() {
    if (typeof Chart === 'undefined') return;
    Chart.defaults.font.size = 14;
    if (Chart.defaults.plugins.legend && Chart.defaults.plugins.legend.labels) {
        Chart.defaults.plugins.legend.labels.font = { size: 14 };
    }
    if (Chart.defaults.plugins.title) {
        Chart.defaults.plugins.title.font = { size: 16, weight: '600' };
    }
}
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', _applyChartDefaults);
} else {
    _applyChartDefaults();
}

// ============================================================================
// SHARED LEGEND UTILITY - buildLegend()
// ============================================================================
// Generates consistent HTML legends with correct swatch types for all pages.
// Swatch types: 'line', 'band', 'dashed', 'dot-line', 'hatch'
//
// Usage:
//   buildLegend(container, [
//     { label: 'Solar', color: RESOURCE_COLORS.solar, type: 'band' },
//     { label: 'Demand', color: '#1A2744', type: 'dashed' },
//     { label: 'Existing', color: '#888', type: 'line' },
//   ]);
//
// Or inject as innerHTML:
//   el.innerHTML = buildLegendHTML([...items]);
// ============================================================================

function buildLegendHTML(items) {
    return items.map(function(item) {
        var type = item.type || 'band';
        var cls = 'swatch-' + type;
        var style = '';
        if (type === 'line' || type === 'dot-line') {
            style = 'background:' + item.color;
        } else if (type === 'band') {
            style = 'background:' + item.color;
        } else if (type === 'dashed') {
            style = 'border-color:' + item.color;
        } else if (type === 'hatch') {
            style = 'color:' + item.color;
        }
        return '<span class="chart-legend-item">' +
            '<span class="' + cls + '" style="' + style + '"></span>' +
            item.label + '</span>';
    }).join('');
}

function buildLegend(container, items) {
    if (typeof container === 'string') {
        container = document.getElementById(container);
    }
    if (!container) return;
    container.className = (container.className || '').indexOf('chart-legend') >= 0
        ? container.className
        : (container.className + ' chart-legend').trim();
    container.innerHTML = buildLegendHTML(items);
}

// Helper: build legend items from a Chart.js chart instance
// Automatically detects line vs bar datasets and uses correct swatch types
function buildLegendFromChart(container, chart, options) {
    options = options || {};
    var filter = options.filter || function() { return true; };
    var items = [];
    chart.data.datasets.forEach(function(ds, i) {
        if (!filter(ds, i)) return;
        var type = 'band';
        if (ds.type === 'line' || (!ds.type && chart.config.type === 'line')) {
            if (ds.borderDash && ds.borderDash.length) {
                type = 'dashed';
            } else if (ds.fill === false || ds.fill === undefined) {
                type = 'line';
            }
        }
        items.push({
            label: ds.label,
            color: ds.borderColor || ds.backgroundColor,
            type: type
        });
    });
    buildLegend(container, items);
}
