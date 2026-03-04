// ============================================================================
// CHART COLORS — Canonical color constants for Chart.js
// ============================================================================
// Single source of truth for all chart colors. NEVER hardcode hex values
// in Chart.js dataset configurations — always use these constants.
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
    battery:      '#8B5CF6',
    ldes:         '#E91E63',
    greenH2:      '#10B981',
    geothermal:   '#B45309',
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
    batteryT:      'rgba(139, 92, 246, 0.55)',
    ldesT:         'rgba(233, 30, 99, 0.55)',
    greenH2T:      'rgba(16, 185, 129, 0.55)',
    geothermalT:   'rgba(180, 83, 9, 0.55)',
    storageT:      'rgba(239, 68, 68, 0.55)',
    gapT:          'rgba(209, 213, 219, 0.55)',

    // Light backgrounds (8% opacity)
    solarBg:        'rgba(245, 158, 11, 0.08)',
    windBg:         'rgba(34, 197, 94, 0.08)',
    offshoreWindBg: 'rgba(0, 150, 136, 0.08)',
    hydroBg:        'rgba(14, 165, 233, 0.08)',
    nuclearBg:      'rgba(99, 102, 241, 0.08)',
    ccsBg:          'rgba(100, 116, 139, 0.08)',
    batteryBg:      'rgba(139, 92, 246, 0.08)',
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
    MISO:  '#06B6D4',
    SPP:   '#A855F7',

    // Transparent fills (12% opacity)
    CAISO_T: 'rgba(245, 158, 11, 0.12)',
    ERCOT_T: 'rgba(34, 197, 94, 0.12)',
    PJM_T:   'rgba(14, 165, 233, 0.12)',
    NYISO_T: 'rgba(233, 30, 99, 0.12)',
    NEISO_T: 'rgba(156, 39, 176, 0.12)',
    MISO_T:  'rgba(6, 182, 212, 0.12)',
    SPP_T:   'rgba(168, 85, 247, 0.12)',

    // Ordered list (for iteration)
    list: ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP'],
    colors: ['#F59E0B', '#22C55E', '#0EA5E9', '#E91E63', '#9C27B0', '#06B6D4', '#A855F7'],
    fills:  [
        'rgba(245, 158, 11, 0.12)', 'rgba(34, 197, 94, 0.12)',
        'rgba(14, 165, 233, 0.12)', 'rgba(233, 30, 99, 0.12)',
        'rgba(156, 39, 176, 0.12)', 'rgba(6, 182, 212, 0.12)',
        'rgba(168, 85, 247, 0.12)'
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

// Resource stack order (for stacked charts — bottom to top)
var RESOURCE_STACK_ORDER = ['solar', 'wind', 'offshoreWind', 'hydro', 'nuclear', 'ccs', 'battery', 'ldes', 'greenH2', 'geothermal', 'gap'];

// ============================================================================
// SHARED LEGEND UTILITY — buildLegend()
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
