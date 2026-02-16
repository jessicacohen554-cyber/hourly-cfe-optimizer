// ============================================================================
// SHARED DATA MODULE — Single source of truth for all dashboard pages
// ============================================================================
// RULE: No data constants defined in HTML files. Change here, propagates everywhere.
// Updated: 2026-02-16 from tranche-repriced results (Step 2 + Step 3 pipeline)
// ============================================================================

// --- Thresholds (from optimizer) ---
const THRESHOLDS = [75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99];

// --- Average MAC ($/ton CO2) — from Option B statistical pipeline ---
// Source: compute_mac_stats.py using per-scenario CO2 (fuel-switching elasticity)
// CO2 methodology: hourly fossil-fuel emission rates with fuel-switching shifts
// Medium = monotonic envelope (running max of MMM_M_M scenario)
// Low/High = P10/P90 from 324-scenario factorial experiment
const MAC_DATA = {
    medium: {
        CAISO:  [104, 104, 104, 104, 106, 110, 113, 115, 117],
        ERCOT:  [27,  29,  32,  36,  36,  40,  43,  46,  50],
        PJM:    [109, 109, 112, 112, 112, 112, 112, 112, 112],
        NYISO:  [171, 171, 171, 171, 171, 171, 171, 171, 171],
        NEISO:  [106, 114, 117, 117, 118, 122, 123, 159, 169]
    },
    low: {
        CAISO:  [50,  51,  53,  53,  54,  57,  61,  68,  68],
        ERCOT:  [12,  12,  15,  15,  18,  21,  24,  28,  30],
        PJM:    [55,  55,  55,  55,  55,  55,  55,  55,  55],
        NYISO:  [73,  73,  79,  82,  83,  83,  83,  94,  94],
        NEISO:  [44,  52,  56,  59,  61,  65,  71,  80,  88]
    },
    high: {
        CAISO:  [114, 121, 126, 127, 132, 135, 136, 141, 149],
        ERCOT:  [40,  43,  46,  49,  50,  53,  58,  62,  66],
        PJM:    [132, 132, 132, 132, 132, 132, 132, 132, 132],
        NYISO:  [190, 190, 190, 190, 190, 190, 190, 190, 190],
        NEISO:  [154, 155, 157, 159, 160, 166, 170, 180, 189]
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
// Medium = MMM_M_M scenario with convex hull correction
// Low/High = P10/P90 across 324 scenarios with monotonicity enforcement
// Cap: $1000/ton (NREL literature max for sub-100% steps)
const MARGINAL_MAC_LABELS = ['75→90%', '90→92.5%', '92.5→95%', '95→97.5%', '97.5→99%'];

const MARGINAL_MAC_DATA = {
    medium: {
        CAISO:  [237, 539, 539, 872, 872],
        ERCOT:  [164, 164, 174, 183, 316],
        PJM:    [155, 155, 930, 930, 930],
        NYISO:  [367, 367, 367, 367, 367],
        NEISO:  [284, 355, 355, 1000, 1000]
    },
    low: {
        CAISO:  [100, 75,  123, 158, 214],
        ERCOT:  [60,  100, 121, 112, 169],
        PJM:    [106, 124, 481, 71,  299],
        NYISO:  [106, 218, 143, 210, 307],
        NEISO:  [72,  269, 85,  339, 390]
    },
    high: {
        CAISO:  [1000, 1000, 389, 1000, 472],
        ERCOT:  [370, 203, 271, 264, 385],
        PJM:    [389, 660, 1000, 541, 1000],
        NYISO:  [907, 991, 488, 654, 1000],
        NEISO:  [421, 557, 752, 1000, 1000]
    }
};

// --- Effective Cost per Useful MWh ($/MWh) ---
// Source: Step 2 cost optimization (tranche-repriced MMM_M_M scenario) + Step 3 postprocess
// Merit-order tranche pricing: nuclear uprates (5% of fleet, capped) filled first,
// then regional new-build (geothermal CAISO, SMR elsewhere)
// Monotonicity enforced via running max (45Q correction can create artifacts)
// Indices match THRESHOLDS array: [75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99]
const EFFECTIVE_COST_DATA = {
    CAISO:  [59.2, 59.2, 61.3, 63.0, 64.8, 66.6, 69.2, 70.0, 71.4],
    ERCOT:  [38.5, 39.6, 41.7, 43.7, 43.9, 46.6, 48.8, 50.7, 53.5],
    PJM:    [82.0, 85.0, 89.4, 89.4, 89.4, 89.4, 89.4, 89.4, 89.4],
    NYISO:  [96.5, 96.5, 99.4, 99.4, 99.4, 99.4, 99.4, 103.9, 103.9],
    NEISO:  [75.3, 79.6, 82.5, 83.5, 84.9, 87.2, 87.9, 101.8, 108.6]
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
