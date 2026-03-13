/**
 * Demo Engine — Client-side synthetic market simulation
 * Generates plausible results from L/M/H price level + carbon price + ISO.
 * NOT a real model — directionally correct for prototype feedback.
 */

const ISO_PROFILES = {
    CAISO: { demand_twh: 224, clean_pct: 48.5, fossil_gw: 47, nuc_gw: 2.3, coal_gw: 0, gas_cc_gw: 25, gas_ct_gw: 15, oil_gw: 0.5, cap_market: 75, base_lmp: 52 },
    ERCOT: { demand_twh: 488, clean_pct: 46.1, fossil_gw: 80, nuc_gw: 5.1, coal_gw: 14, gas_cc_gw: 42, gas_ct_gw: 18, oil_gw: 1, cap_market: 0, base_lmp: 38 },
    PJM:   { demand_twh: 843, clean_pct: 40.6, fossil_gw: 128, nuc_gw: 33, coal_gw: 28, gas_cc_gw: 55, gas_ct_gw: 30, oil_gw: 5, cap_market: 120, base_lmp: 45 },
    NYISO: { demand_twh: 152, clean_pct: 39.0, fossil_gw: 28, nuc_gw: 3.4, coal_gw: 0.5, gas_cc_gw: 14, gas_ct_gw: 10, oil_gw: 2, cap_market: 85, base_lmp: 48 },
    NEISO: { demand_twh: 115, clean_pct: 33.5, fossil_gw: 16, nuc_gw: 3.3, coal_gw: 0.3, gas_cc_gw: 9, gas_ct_gw: 5, oil_gw: 0.8, cap_market: 55, base_lmp: 55 },
    MISO:  { demand_twh: 660, clean_pct: 31.3, fossil_gw: 105, nuc_gw: 12, coal_gw: 42, gas_cc_gw: 38, gas_ct_gw: 20, oil_gw: 2, cap_market: 25, base_lmp: 35 },
    SPP:   { demand_twh: 296, clean_pct: 47.0, fossil_gw: 58, nuc_gw: 1.2, coal_gw: 18, gas_cc_gw: 22, gas_ct_gw: 12, oil_gw: 1, cap_market: 0, base_lmp: 32 },
};

const PRICE_LEVELS = {
    Low:    { gas: 2.00, coal: 2.00, oil: 8.00 },
    Medium: { gas: 3.50, coal: 2.25, oil: 10.50 },
    High:   { gas: 6.00, coal: 2.50, oil: 13.00 },
};

function runDemoSimulation(iso, priceLevel, carbonPrice) {
    const p = ISO_PROFILES[iso];
    const fuel = PRICE_LEVELS[priceLevel];
    if (!p || !fuel) return null;

    // Marginal costs by generator type
    const mc_coal = fuel.coal * 10.0 + 5.50 + carbonPrice * 0.95;
    const mc_gas_cc = fuel.gas * 7.0 + 3.50 + carbonPrice * 0.37;
    const mc_gas_ct = fuel.gas * 10.5 + 5.00 + carbonPrice * 0.55;
    const mc_oil_ct = fuel.oil * 10.5 + 6.00 + carbonPrice * 0.65;

    // Average LMP driven by marginal gas CC (price-setter in most hours)
    const avg_lmp = mc_gas_cc * 0.85 + mc_gas_ct * 0.15 + (p.cap_market > 0 ? 3 : 8);

    // Higher carbon → more coal retirement → higher clean %
    const coal_retirement_frac = Math.min(1, carbonPrice / 80);
    const gas_ct_retirement_frac = Math.min(0.4, Math.max(0, (carbonPrice - 40) / 150));
    const clean_boost = coal_retirement_frac * (p.coal_gw / p.fossil_gw) * 30 +
                        gas_ct_retirement_frac * (p.gas_ct_gw / p.fossil_gw) * 10;
    const clean_pct = Math.min(85, p.clean_pct + clean_boost + (priceLevel === 'High' ? -3 : priceLevel === 'Low' ? 4 : 0));

    // New clean capacity needed
    const new_gw = Math.max(0, (clean_pct - p.clean_pct) / 100 * p.fossil_gw * 0.4);

    // Generator economics
    const gens = [];
    function addGen(type, cap_gw, mc, hr, co2) {
        if (cap_gw <= 0.01) return;
        const cf = Math.max(0.05, Math.min(0.92, (avg_lmp - mc) / avg_lmp + 0.4));
        const dispatch_hrs = Math.round(cf * 8760);
        const profit = avg_lmp * cf - mc;
        let status = 'profitable';
        if (profit < -5) status = 'retiring';
        else if (profit < 5) status = 'marginal';
        gens.push({
            unit_type: type,
            capacity_mw: Math.round(cap_gw * 1000),
            marginal_cost: +mc.toFixed(2),
            dispatch_hours: dispatch_hrs,
            capacity_factor: +cf.toFixed(3),
            avg_revenue_mwh: +(avg_lmp * cf).toFixed(2),
            profit_mwh: +profit.toFixed(2),
            status: status,
        });
    }

    addGen('coal_steam', p.coal_gw * (1 - coal_retirement_frac), mc_coal, 10.0, 0.95);
    addGen('gas_ccgt', p.gas_cc_gw, mc_gas_cc, 7.0, 0.37);
    addGen('gas_ct', p.gas_ct_gw * (1 - gas_ct_retirement_frac), mc_gas_ct, 10.5, 0.55);
    addGen('oil_ct', p.oil_gw, mc_oil_ct, 10.5, 0.65);

    // Nuclear revenue
    const nuc_energy = avg_lmp * 0.93; // 93% CF
    const nuc_cap = p.cap_market > 0 ? p.cap_market / 8.76 : 0; // $/kW-yr → $/MWh
    const nuc_ptc = 15; // PTC value
    const nuc_total = nuc_energy + nuc_cap + nuc_ptc;

    // CCS breakeven
    const ccs_lcoe = 86; // Medium CCS
    const ccs_breakeven = Math.max(0, (ccs_lcoe - mc_gas_cc) / (0.37 - 0.037));

    // What gets built (resource shares for new capacity)
    const solar_share = iso === 'CAISO' ? 0.40 : 0.30;
    const wind_share = (iso === 'ERCOT' || iso === 'SPP' || iso === 'MISO') ? 0.35 : 0.20;
    const nuc_share = 0.15;
    const storage_share = 1 - solar_share - wind_share - nuc_share;

    // LMP by clean threshold
    const thresholds = [30, 40, 50, 60, 70, 80, 90, 95, 99];
    const lmp_by_threshold = {};
    thresholds.forEach(t => {
        const suppression = (t - p.clean_pct) * 0.35;
        lmp_by_threshold[t] = { avg_lmp: Math.max(15, avg_lmp - suppression) };
    });

    // CCS analysis
    const carbon_prices_sweep = [0, 25, 50, 75, 100, 125, 150, 175, 200];
    const ccs_analysis = {
        carbon_prices: carbon_prices_sweep,
        existing_ccgt_cost: carbon_prices_sweep.map(cp => fuel.gas * 7.0 + 3.50 + cp * 0.37),
        ccs_retrofit_cost: carbon_prices_sweep.map(cp => 65 + cp * 0.037),
        new_gas_cost: carbon_prices_sweep.map(cp => fuel.gas * 6.5 + 4.00 + cp * 0.37),
    };

    // Gas fleet shift
    const gas_fleet_shift = carbon_prices_sweep.map(cp => ({
        carbon_price: cp,
        efficient_cf: Math.min(0.92, 0.65 + cp * 0.002),
        avg_cf: Math.max(0.15, 0.55 - cp * 0.002),
        old_cf: Math.max(0.05, 0.35 - cp * 0.003),
    }));

    // Sensitivity matrix (gas × carbon)
    const gas_prices = [2, 3, 4, 5, 6];
    const carbon_prices_sens = [0, 25, 50, 75, 100];
    const sens_values = carbon_prices_sens.map(cp =>
        gas_prices.map(gp => {
            const mc = gp * 7.0 + 3.50 + cp * 0.37;
            return +(p.clean_pct + (mc - 28) * 0.8).toFixed(1);
        })
    );

    return {
        iso: iso,
        mode: 'snapshot',
        market_outcome_clean_pct: +clean_pct.toFixed(1),
        avg_lmp: +avg_lmp.toFixed(2),
        new_capacity_gw: +new_gw.toFixed(1),
        nuclear_revenue: {
            energy_rev_mwh: +nuc_energy.toFixed(2),
            capacity_rev_mwh: +nuc_cap.toFixed(2),
            ptc_mwh: nuc_ptc,
            total_mwh: +nuc_total.toFixed(2),
        },
        ccs_breakeven_carbon_price: +ccs_breakeven.toFixed(1),
        generator_economics: gens,
        what_gets_built: {
            solar: +(new_gw * solar_share).toFixed(1),
            wind: +(new_gw * wind_share).toFixed(1),
            nuclear: +(new_gw * nuc_share).toFixed(1),
            battery: +(new_gw * storage_share).toFixed(1),
        },
        threshold_sweep: lmp_by_threshold,
        ccs_analysis: ccs_analysis,
        gas_fleet_shift: gas_fleet_shift,
        sensitivity_matrix: {
            gas_prices: gas_prices,
            carbon_prices: carbon_prices_sens,
            values: sens_values,
        },
    };
}
