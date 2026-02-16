// ============================================================================
// SHARED DATA MODULE — Single source of truth for all dashboard pages
// ============================================================================
// RULE: No data constants defined in HTML files. Change here, propagates everywhere.
// Updated: 2026-02-15 from optimizer results (overprocure_results.json)
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
        CAISO:  [93,  96,  104, 104, 107, 107, 109, 116, 118],
        ERCOT:  [24,  26,  31,  35,  36,  40,  43,  46,  50],
        PJM:    [49,  49,  49,  50,  52,  53,  56,  63,  63],
        NYISO:  [104, 108, 108, 108, 108, 108, 108, 120, 120],
        NEISO:  [115, 115, 122, 123, 125, 129, 137, 146, 146]
    },
    low: {
        CAISO:  [46,  48,  50,  52,  53,  56,  59,  65,  66],
        ERCOT:  [5,   6,   10,  13,  14,  17,  20,  23,  25],
        PJM:    [12,  13,  14,  16,  17,  18,  20,  24,  27],
        NYISO:  [50,  49,  50,  51,  51,  55,  57,  70,  75],
        NEISO:  [52,  58,  61,  61,  63,  67,  72,  76,  84]
    },
    high: {
        CAISO:  [110, 120, 122, 127, 126, 132, 139, 147, 153],
        ERCOT:  [39,  41,  46,  49,  52,  54,  59,  63,  68],
        PJM:    [65,  68,  72,  74,  75,  78,  79,  87,  92],
        NYISO:  [139, 146, 148, 150, 153, 160, 166, 175, 182],
        NEISO:  [145, 145, 147, 147, 150, 153, 165, 175, 183]
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
        Low:    { low: 250,  mid: 400,  high: 600 },
        Medium: { low: 400,  mid: 600,  high: 1500 },
        High:   { low: 600,  mid: 1000, high: 1500 },
        sources: 'ETH Zurich 2024, Climeworks, Belfer Center'
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

// --- Stepwise Marginal MAC ($/ton CO2) — median (P50) across 324 scenarios ---
// Source: compute_mac_stats.py — P50 of (delta_cost × demand) / delta_co2 per scenario
// CO2 varies by scenario (fuel-switching elasticity on marginal emission rates)
// Index 0 = 75% level (no prior step), indices 1-9 = steps 75→80, ..., 99→100
const MARGINAL_MAC_DATA = {
    medium: {
        CAISO:  [null, 214, 116, 475, 138, 290, 305, 347, 340],
        ERCOT:  [null,  78, 112, 118, 110, 154, 192, 208, 266],
        PJM:    [null,  62,  63,  82, 105, 150, 339, 215, 514],
        NYISO:  [null, 117,  74,  98, 155, 260, 310, 449, 399],
        NEISO:  [null, 141, 166, 108, 170, 329, 536, 491, 677]
    }
};

// --- Crossover Summary (from stepwise MAC P50) ---
const CROSSOVER_SUMMARY = {
    // Threshold at which stepwise marginal MAC first exceeds SCC ($190/ton)
    scc_190: { CAISO: 75, ERCOT: 95, PJM: 95, NYISO: 87.5, NEISO: 80 },
    // Threshold at which stepwise marginal MAC first exceeds DAC low ($400/ton)
    dac_400: { CAISO: 85, ERCOT: '>99', PJM: 97.5, NYISO: 95, NEISO: 92.5 },
    // Threshold at which stepwise marginal MAC first exceeds DAC high ($600/ton)
    dac_600: { CAISO: '>99', ERCOT: '>99', PJM: 99, NYISO: '>99', NEISO: 95 }
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
 * Returns the "from" threshold of that step, or '>99' if never exceeded.
 * Skips null entries (CO2 decreased or no change at that step).
 */
function findMarginalCrossover(regionMarginals, costLevel) {
    for (let i = 1; i < regionMarginals.length; i++) {
        if (regionMarginals[i] !== null && regionMarginals[i] > costLevel) {
            return THRESHOLDS[i - 1]; // "from" threshold of this step
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
