// ============================================================================
// SHARED DATA MODULE — Single source of truth for all dashboard pages
// ============================================================================
// RULE: No data constants defined in HTML files. Change here, propagates everywhere.
// Updated: 2026-02-16 from simplified 2-tranche CF repricing (Step 2 delta + Step 3 pipeline)
// ============================================================================

// --- Thresholds (from optimizer) ---
const THRESHOLDS = [75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99];

// --- Average MAC ($/ton CO2) — from Option B statistical pipeline ---
// Source: compute_mac_stats.py using per-scenario CO2 (fuel-switching elasticity)
// CO2 methodology: hourly fossil-fuel emission rates with fuel-switching shifts
// Medium = monotonic envelope (top-down ceiling of MMM_M_M scenario)
// Low/High = P10/P90 from 324-scenario factorial experiment (ceiling-enforced)
const MAC_DATA = {
    medium: {
        CAISO:  [93,  95,  103, 104, 106, 110, 113, 115, 118],
        ERCOT:  [24,  26,  31,  35,  36,  40,  43,  46,  50],
        PJM:    [67,  77,  89,  91,  93,  95,  98,  100, 100],
        NYISO:  [123, 123, 138, 138, 139, 139, 148, 156, 156],
        NEISO:  [106, 111, 116, 116, 118, 122, 128, 157, 169]
    },
    low: {
        CAISO:  [46,  47,  50,  52,  53,  56,  60,  66,  66],
        ERCOT:  [5,   6,   10,  13,  14,  17,  20,  22,  25],
        PJM:    [24,  30,  32,  36,  38,  41,  42,  49,  53],
        NYISO:  [60,  67,  74,  75,  78,  83,  86,  94,  101],
        NEISO:  [44,  52,  55,  59,  61,  65,  70,  80,  88]
    },
    high: {
        CAISO:  [114, 120, 123, 127, 128, 135, 138, 146, 153],
        ERCOT:  [39,  41,  46,  49,  52,  55,  59,  63,  69],
        PJM:    [86,  89,  93,  96,  98,  102, 107, 112, 113],
        NYISO:  [146, 146, 151, 151, 152, 159, 165, 177, 182],
        NEISO:  [151, 151, 157, 158, 160, 166, 169, 181, 189]
    }
};

// --- Region Colors (used by abatement pages) ---
const REGION_COLORS = {
    CAISO: '#F59E0B',
    ERCOT: '#22C55E',
    PJM:   '#4a90d9',
    NYISO: '#E91E63',
    NEISO: '#9C27B0'
};

// --- Resource Colors & Labels (used by dashboard, index, region_deepdive) ---
const MIX_RESOURCES = ['clean_firm', 'ccs_ccgt', 'solar', 'wind', 'hydro', 'battery', 'ldes'];

const MIX_LABELS_MAP = {
    clean_firm: 'Clean Firm',
    ccs_ccgt:   'CCS-CCGT',
    solar:      'Solar',
    wind:       'Wind',
    hydro:      'Hydro',
    battery:    'Battery (4hr)',
    ldes:       'LDES (100hr)'
};

const MIX_COLORS = {
    clean_firm: { fill: 'rgba(30,58,95,0.50)',    border: '#1E3A5F' },
    ccs_ccgt:   { fill: 'rgba(13,148,136,0.50)',  border: '#0D9488' },
    solar:      { fill: 'rgba(245,158,11,0.50)',  border: '#F59E0B' },
    wind:       { fill: 'rgba(34,197,94,0.50)',   border: '#22C55E' },
    hydro:      { fill: 'rgba(14,165,233,0.50)',  border: '#0EA5E9' },
    battery:    { fill: 'rgba(139,92,246,0.50)',  border: '#8B5CF6' },
    ldes:       { fill: 'rgba(236,72,153,0.50)',  border: '#EC4899' }
};

// --- Benchmark Data (static — researched L/M/H with sources) ---
const BENCHMARKS_STATIC = [
    { name: 'Energy Efficiency (Buildings)', short: 'Energy Efficiency', low: -100, mid: 0,   high: 60,   color: '#4CAF50', category: 'demand_reduction', trajectory: 'stable', confidence: 'high',
      sources: 'EDF/Evolved Energy MAC 2.0, World Bank, Gillingham & Stock' },
    { name: 'EU ETS Price',                  short: 'EU ETS', low: 65,   mid: 88,  high: 92,   color: '#2196F3', category: 'benchmark', trajectory: 'rising', confidence: 'high',
      sources: 'Trading Economics, Sandbag, BNEF' },
    { name: 'SCC \u2014 EPA ($190/ton)',     short: 'SCC (EPA)', low: 140,  mid: 190, high: 380,  color: '#FF9800', category: 'benchmark', trajectory: 'rising', confidence: 'medium',
      sources: 'EPA Dec 2023 Report' },
    { name: 'SCC \u2014 Rennert et al.',     short: 'SCC (Rennert)', low: 120,  mid: 185, high: 450,  color: '#E65100', category: 'benchmark', trajectory: 'rising', confidence: 'medium',
      sources: 'Rennert et al. (2022) Nature' },
    { name: 'Carbon Credits (Nature)',       short: 'Carbon Credits', low: 3,    mid: 15,  high: 35,   color: '#9E9E9E', category: 'voluntary', trajectory: 'rising', confidence: 'medium',
      sources: 'Sylvera 2026, Regreener, MSCI' }
];

// --- Benchmark Data (dynamic — shift with user toggles) ---
const BENCHMARKS_DYNAMIC = {
    dac: {
        name: 'Direct Air Capture (DAC)', short: 'DAC', color: '#E91E63', category: 'carbon_removal', confidence: 'low',
        trajectory: 'declining_steep',
        // 2040-2045 projected costs with learning curves
        // Sources: DOE Liftoff NOAK, IEAGHG 2021, Fasihi et al. (2019), Sievert/McQueen (Joule 2024),
        //          Climeworks Gen 3 roadmap, DOE Carbon Negative Shot, Kanyako & Craig (Earth's Future 2025)
        // Low = DOE NOAK + aggressive learning (12-15% LR); Mid = IEAGHG NOAK + moderate learning (8-10% LR);
        // High = Sievert et al. multi-component curves + slow deployment (5-8% LR)
        Low:    { low: 65,   mid: 100,  high: 175 },
        Medium: { low: 100,  mid: 175,  high: 300 },
        High:   { low: 175,  mid: 300,  high: 500 },
        sources: 'DOE Liftoff NOAK, IEAGHG 2021, Fasihi et al. (J. Cleaner Prod. 2019), Sievert et al. (Joule 2024), Climeworks Gen 3 roadmap, DOE Carbon Negative Shot, Kanyako & Craig (Earth\'s Future 2025)'
    },
    industrial: {
        name: 'Industrial Electrification', short: 'Ind. Electrification', color: '#8BC34A', category: 'industrial_decarb', confidence: 'medium',
        trajectory: 'declining',
        Low:    { low: -50, mid: 20,  high: 60 },
        Medium: { low: 0,   mid: 60,  high: 160 },
        High:   { low: 60,  mid: 120, high: 250 },
        sources: 'McKinsey, Thunder Said Energy, Rewiring America'
    },
    removal: {
        name: 'Carbon Removal (BECCS + Enhanced Weathering)', short: 'Carbon Removal\u00B9', color: '#009688', category: 'carbon_removal', confidence: 'low',
        trajectory: 'declining',
        Low:    { low: 20,  mid: 75,  high: 150 },
        Medium: { low: 50,  mid: 150, high: 300 },
        High:   { low: 100, mid: 200, high: 350 },
        sources: 'ORNL, Nature 2024, Nature Comms 2025'
    }
};

// --- Extra Benchmarks (not toggle-controlled) ---
const BENCHMARKS_EXTRA = [
    { name: 'Green Hydrogen (Industrial)',  short: 'Green H\u2082', low: 150, mid: 500, high: 1250, color: '#00BCD4', category: 'industrial_decarb',
      trajectory: 'declining', confidence: 'low', sources: 'Shafiee & Schrag (Joule 2024), Belfer Center' },
    { name: 'Sustainable Aviation Fuel',    short: 'SAF', low: 136, mid: 300, high: 500,  color: '#FF5722', category: 'transport',
      trajectory: 'declining', confidence: 'medium', sources: 'NREL, RMI 2025, ICCT, WEF' },
    { name: 'CDR Credits (Engineered)',     short: 'CDR Credits', low: 177, mid: 320, high: 600,  color: '#7C4DFF', category: 'voluntary',
      trajectory: 'declining', confidence: 'medium', sources: 'Sylvera, CarbonCredits.com' },
    { name: 'EVs vs ICE (Fleet)',           short: 'EVs vs ICE', low: -50, mid: 250, high: 970,  color: '#FF6F00', category: 'transport',
      trajectory: 'declining_steep', confidence: 'medium', sources: 'Argonne Labs, Penn Wharton, RFF' }
];

// --- Two-Zone Marginal MAC ($/ton CO2) ---
// Zone 1 (75→90%): single aggregate MAC — grid backbone cost per ton
// Zone 2 (90→99%): granular steps with enforced monotonicity (non-decreasing)
// Medium = MMM_M_M scenario stepwise envelope with top-down ceiling
// Low/High = P10/P90 across 324 scenarios with top-down ceiling enforcement
// Cap: $1000/ton (NREL literature max for sub-100% steps)
const MARGINAL_MAC_LABELS = ['75→90%', '90→92.5%', '92.5→95%', '95→97.5%', '97.5→99%'];

const MARGINAL_MAC_DATA = {
    medium: {
        CAISO:  [239, 517, 517, 1000, 1000],
        ERCOT:  [162, 208, 208, 208, 302],
        PJM:    [270, 281, 512, 512, 512],
        NYISO:  [215, 257, 336, 336, 336],
        NEISO:  [223, 369, 683, 1000, 1000]
    },
    low: {
        CAISO:  [65,  81,  135, 160, 236],
        ERCOT:  [63,  94,  114, 115, 198],
        PJM:    [68,  145, 149, 149, 233],
        NYISO:  [89,  149, 149, 149, 149],
        NEISO:  [60,  99,  99,  311, 390]
    },
    high: {
        CAISO:  [493, 493, 493, 493, 493],
        ERCOT:  [185, 210, 265, 272, 388],
        PJM:    [254, 416, 416, 416, 865],
        NYISO:  [417, 462, 462, 659, 834],
        NEISO:  [398, 557, 738, 1000, 1000]
    }
};

// --- Effective Cost per Useful MWh ($/MWh) ---
// Source: Step 2 cost optimization (tranche-repriced MMM_M_M scenario) + Step 3 postprocess
// Merit-order tranche pricing: nuclear uprates (5% of fleet, capped) filled first,
// then regional new-build (geothermal CAISO, SMR elsewhere)
// Monotonicity enforced via top-down ceiling (lower thresholds capped to next higher)
// Indices match THRESHOLDS array: [75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99]
const EFFECTIVE_COST_DATA = {
    CAISO:  [55.9, 57.0, 61.3, 63.0, 64.9, 66.6, 69.2, 70.2, 71.7],
    ERCOT:  [36.9, 38.1, 41.1, 43.7, 44.4, 46.6, 48.8, 50.8, 53.5],
    PJM:    [63.6, 69.9, 78.0, 81.4, 83.1, 84.4, 86.8, 89.1, 89.1],
    NYISO:  [81.2, 82.2, 94.1, 94.7, 95.2, 95.2, 101.3, 104.1, 105.7],
    NEISO:  [75.1, 78.9, 82.2, 83.5, 84.9, 87.2, 89.8, 101.3, 108.8]
};

// --- Nuclear Uprate Caps (TWh/yr) — 5% of existing nuclear at 90% CF ---
const UPRATE_CAPS_TWH = {
    CAISO: 0.907, ERCOT: 1.064, PJM: 12.614, NYISO: 1.340, NEISO: 1.380
};

// ============================================================================
// SHARED UTILITY FUNCTIONS
// ============================================================================

/**
 * Find the threshold at which regionData first exceeds costLevel.
 * @param {number[]} regionData - MAC values at each threshold
 * @param {number} costLevel - benchmark cost to compare against
 * @returns {number|string} threshold value or '>99'
 */
function findCrossover(regionData, costLevel) {
    for (let i = 0; i < regionData.length; i++) {
        if (regionData[i] >= costLevel) return THRESHOLDS[i];
    }
    return '>99';
}

/**
 * Find the threshold step at which MARGINAL MAC first exceeds a benchmark cost.
 * Uses two-zone format: index 0 = 75→90%, indices 1-4 = zone 2 steps.
 * Returns the "from" threshold of that step, or '>99' if never exceeded.
 */
const MARGINAL_THRESHOLDS = [75, 90, 92.5, 95, 97.5];
function findMarginalCrossover(regionMarginals, costLevel) {
    for (let i = 0; i < regionMarginals.length; i++) {
        if (regionMarginals[i] !== null && regionMarginals[i] > costLevel) {
            return MARGINAL_THRESHOLDS[i];
        }
    }
    return '>99';
}

/**
 * CSS class for inflection table cells based on crossover threshold.
 */
function cellClass(val) {
    if (val === '>99' || val === '>100') return 'cell-green';
    if (val >= 97) return 'cell-green';
    if (val >= 95) return 'cell-yellow';
    if (val >= 92) return 'cell-orange';
    return 'cell-red';
}

/**
 * Get all benchmarks merged and sorted, using specified toggle state.
 * @param {Object} state - { dac: 'Medium', industrial: 'Medium', removal: 'Medium' }
 * @returns {Array} sorted benchmark objects
 */
function getAllBenchmarks(state) {
    state = state || { dac: 'Medium', industrial: 'Medium', removal: 'Medium' };
    const dynamic = Object.keys(BENCHMARKS_DYNAMIC).map(key => {
        const b = BENCHMARKS_DYNAMIC[key];
        const costs = b[state[key]] || b.Medium;
        return { name: b.name, short: b.short || b.name, low: costs.low, mid: costs.mid, high: costs.high,
                 color: b.color, category: b.category, confidence: b.confidence, trajectory: b.trajectory, sources: b.sources };
    });
    return [...BENCHMARKS_STATIC, ...dynamic, ...BENCHMARKS_EXTRA]
        .filter(b => b.mid >= 0)
        .sort((a, b) => a.mid - b.mid);
}
