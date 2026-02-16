// ============================================================================
// SHARED DATA MODULE — Single source of truth for all dashboard pages
// ============================================================================
// RULE: No data constants defined in HTML files. Change here, propagates everywhere.
// Updated: 2026-02-15 from optimizer results (overprocure_results.json)
// ============================================================================

// --- Thresholds (from optimizer) ---
const THRESHOLDS = [75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99];

// --- Average MAC ($/ton CO2) — monotonicity-enforced (running max) ---
// Source: incremental_above_baseline * demand_mwh / total_co2_abated_tons
// CO2 methodology: curtailment-aware charge netting (storage charges from clean
// surplus get 0 emission rate; otherwise marginal fossil rate)
// Low/High scaled ×0.72/×1.38 from Medium per NREL ATB cost spread
const MAC_DATA = {
    medium: {
        CAISO:  [87,  87,  95,  96,  98,  98,  103, 107, 109],
        ERCOT:  [23,  25,  29,  34,  34,  36,  39,  42,  47],
        PJM:    [50,  50,  50,  51,  52,  53,  56,  60,  60],
        NYISO:  [107, 107, 107, 107, 107, 107, 108, 119, 119],
        NEISO:  [110, 110, 116, 116, 116, 118, 122, 139, 141]
    },
    low: {
        CAISO:  [63,  63,  68,  69,  71,  71,  74,  77,  79],
        ERCOT:  [16,  18,  21,  24,  25,  26,  28,  30,  34],
        PJM:    [36,  36,  36,  37,  37,  38,  40,  43,  43],
        NYISO:  [77,  77,  77,  77,  77,  77,  78,  86,  86],
        NEISO:  [79,  79,  83,  83,  83,  85,  88,  100, 102]
    },
    high: {
        CAISO:  [121, 121, 131, 132, 135, 135, 142, 148, 151],
        ERCOT:  [31,  34,  40,  47,  47,  49,  54,  58,  65],
        PJM:    [69,  69,  69,  70,  72,  73,  77,  82,  82],
        NYISO:  [148, 148, 148, 148, 148, 148, 149, 165, 165],
        NEISO:  [152, 152, 160, 160, 160, 163, 168, 191, 195]
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

// --- Stepwise Marginal MAC ($/ton CO2) at each threshold step ---
// Source: (delta_incremental_cost * demand_mwh) / delta_co2_abated between adjacent thresholds
// This is the cost of the NEXT increment of CFE matching — the economically meaningful signal
// for "should I push CFE higher or invest in alternative decarb?"
// null = CO2 decreased at that step (resource mix shift) or no additional CO2 abated
const MARGINAL_MAC_DATA = {
    medium: {
        CAISO:  [null,  60, 333, 120, 156,  90, 316, 348, 207],
        ERCOT:  [null,  69,  96, 285,  55,  50, 412, 137, null],
        PJM:    [null,  54,  52,  65,  98, 127, 255, 114, null],
        NYISO:  [null, 113,  53,  73,  94, 324, 131, 887,  56],
        NEISO:  [null,  92, 222,  67, 256, 443, 211, null, 216]
    }
};
// Index 0 = the 75% level itself (no "from" step), indices 1-8 = steps 75→80, 80→85, ..., 97.5→99

// --- Crossover Summary (precomputed from marginal MAC) ---
// "Last efficient threshold" = highest CFE% where ALL prior steps have marginal MAC < benchmark
// These are for narrative use — computed from MARGINAL_MAC_DATA
const CROSSOVER_SUMMARY = {
    // Threshold at which marginal MAC first exceeds SCC ($190/ton)
    scc_190: { CAISO: 80, ERCOT: 85, PJM: 92.5, NYISO: 90, NEISO: 80 },
    // Threshold at which marginal MAC first exceeds DAC low ($400/ton)
    dac_400: { CAISO: '>99', ERCOT: 92.5, PJM: '>99', NYISO: 95, NEISO: 90 },
    // Threshold at which marginal MAC first exceeds DAC high ($600/ton)
    dac_600: { CAISO: '>99', ERCOT: '>99', PJM: '>99', NYISO: 95, NEISO: '>99' }
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
