/**
 * TX ERCOT Real-Fleet Demo Engine
 * Uses actual EIA 860 + 923 generator data (TX_FLEET global)
 * to compute economics under user-selected fuel/carbon price scenarios.
 *
 * Supports:
 * - Per-generator marginal cost, dispatch, profitability
 * - CCGT heat rate binning for stranding analysis
 * - Owner portfolio revenue/profit trajectory
 * - Forward year snapshots (2030-2050) with retirement + learning curves
 */

const FUEL_PRICES = {
    Low:    { gas: 2.00, coal: 1.80, oil: 8.00 },
    Medium: { gas: 3.50, coal: 2.25, oil: 10.50 },
    High:   { gas: 6.00, coal: 2.80, oil: 13.00 },
};

// Nuclear operating cost ($/MWh) - fixed costs amortized
const NUCLEAR_COST_MWH = 29.0;  // Typical all-in operating cost
const NUCLEAR_CF = 0.93;
const NUCLEAR_PTC = 15.0;  // Production tax credit $/MWh (IRA 45U)
const ERCOT_CAP_MARKET = 0;  // ERCOT has no capacity market

// Forward year assumptions
const YEAR_SNAPSHOTS = [2025, 2030, 2035, 2040, 2045, 2050];

// Coal retirement schedule (cumulative fraction retired by year)
const COAL_RETIREMENT_SCHEDULE = {
    2025: 0.00, 2030: 0.20, 2035: 0.50, 2040: 0.80, 2045: 0.95, 2050: 1.00,
};

// Gas plant aging - heat rate degradation per decade
const HR_DEGRADATION_PER_DECADE = 0.15;  // 0.15 MMBtu/MWh per 10 years

// New build costs (LCOE $/MWh) for forward projections - Wright's Law curves
const NEW_BUILD_LCOE = {
    2025: { solar: 32, wind: 28, battery_4hr: 48, nuclear: 88, ccs: 72 },
    2030: { solar: 26, wind: 24, battery_4hr: 38, nuclear: 82, ccs: 65 },
    2035: { solar: 22, wind: 21, battery_4hr: 30, nuclear: 78, ccs: 60 },
    2040: { solar: 19, wind: 19, battery_4hr: 24, nuclear: 74, ccs: 56 },
    2045: { solar: 17, wind: 18, battery_4hr: 20, nuclear: 71, ccs: 53 },
    2050: { solar: 16, wind: 17, battery_4hr: 18, nuclear: 68, ccs: 50 },
};


/**
 * Compute marginal cost for a thermal generator.
 */
function computeMarginalCost(gen, fuelPrices, carbonPrice) {
    const hr = gen.hr || 10.0;
    const vom = gen.vom || 4.0;
    const fuel = gen.fuel;
    let fuelPrice = 0;
    let co2Rate = 0;

    if (fuel === 'coal') {
        fuelPrice = fuelPrices.coal;
        co2Rate = 0.0953;
    } else if (fuel === 'gas') {
        fuelPrice = fuelPrices.gas;
        co2Rate = 0.0531;
    } else if (fuel === 'oil') {
        fuelPrice = fuelPrices.oil;
        co2Rate = 0.0733;
    }

    const fuelCost = fuelPrice * hr;
    const carbonCost = carbonPrice * co2Rate * hr;
    return fuelCost + vom + carbonCost;
}


/**
 * Estimate LMP from merit-order stack (capacity-weighted marginal unit).
 * The price-setting unit is typically the marginal gas CC.
 */
function estimateLMP(generators, fuelPrices, carbonPrice) {
    // Get all thermal generators with costs
    const thermals = generators
        .filter(g => g.fuel === 'coal' || g.fuel === 'gas' || g.fuel === 'oil')
        .map(g => ({
            ...g,
            mc: computeMarginalCost(g, fuelPrices, carbonPrice),
        }))
        .sort((a, b) => a.mc - b.mc);

    if (thermals.length === 0) return 40;

    // Total ERCOT demand ~488 TWh / 8760h ≈ 55.7 GW avg
    // Nuclear + renewables serve ~46% → fossil needs to cover ~30 GW avg
    const totalDemandMW = 55700;
    const cleanSupplyMW = 30000; // Rough: nuclear + wind CF + solar CF in ERCOT
    const fossilNeededMW = totalDemandMW - cleanSupplyMW;

    // Walk up the stack to find the marginal unit
    let cumMW = 0;
    let marginalMC = thermals[0].mc;

    for (const g of thermals) {
        // Derate by typical capacity factor for dispatch
        const availMW = g.mw * 0.85;
        cumMW += availMW;
        marginalMC = g.mc;
        if (cumMW >= fossilNeededMW) break;
    }

    // ERCOT scarcity pricing: add 15% markup for energy-only market
    return marginalMC * 1.08 + 5; // slight markup + scarcity adder
}


/**
 * Run full fleet simulation for a given scenario.
 */
function runFleetSimulation(fuelLevel, carbonPrice, snapshotYear) {
    if (!window.TX_FLEET) return null;

    const fuelPrices = FUEL_PRICES[fuelLevel];
    if (!fuelPrices) return null;

    snapshotYear = snapshotYear || 2025;
    const fleet = TX_FLEET;
    let generators = JSON.parse(JSON.stringify(fleet.generators));

    // ── Forward year adjustments ──
    if (snapshotYear > 2025) {
        const yearsForward = snapshotYear - 2025;

        generators = generators.filter(g => {
            // Coal retirements
            if (g.fuel === 'coal') {
                const retireFrac = COAL_RETIREMENT_SCHEDULE[snapshotYear] || 0;
                // Carbon price accelerates coal retirement
                const carbonAccel = Math.min(0.3, carbonPrice / 150);
                const totalRetire = Math.min(1, retireFrac + carbonAccel);
                // Retire oldest first (by age)
                return Math.random() > totalRetire; // Stochastic for now
            }

            // Old gas CT retirement (age > 40 + carbon pressure)
            if (g.type === 'gas_ct' && g.age && g.age + yearsForward > 45) {
                const retireProb = Math.min(0.8, (g.age + yearsForward - 45) / 20 + carbonPrice / 200);
                return Math.random() > retireProb;
            }

            // Oil CT retirement
            if (g.fuel === 'oil' && g.age && g.age + yearsForward > 35) {
                return Math.random() > 0.7;
            }

            // Check planned retirement
            if (g.retirement) {
                const retYear = parseInt(g.retirement.substring(0, 4));
                if (retYear <= snapshotYear) return false;
            }

            return true;
        });

        // Heat rate degradation for aging gas plants
        generators.forEach(g => {
            if (g.hr && g.fuel === 'gas') {
                g.hr = +(g.hr + (yearsForward / 10) * HR_DEGRADATION_PER_DECADE).toFixed(2);
                g.age = (g.age || 0) + yearsForward;
            }
        });
    }

    // ── Compute per-generator economics ──
    const avgLMP = estimateLMP(generators, fuelPrices, carbonPrice);

    const results = generators.map(g => {
        const isThermal = (g.fuel === 'coal' || g.fuel === 'gas' || g.fuel === 'oil');
        const result = { ...g };

        if (isThermal) {
            const mc = computeMarginalCost(g, fuelPrices, carbonPrice);
            result.mc = +mc.toFixed(2);

            // Dispatch: in-merit if mc < LMP
            if (mc < avgLMP) {
                // CF depends on how far below LMP
                const spread = (avgLMP - mc) / avgLMP;
                if (g.type === 'gas_ccgt' || g.type === 'gas_steam') {
                    result.cf_sim = +Math.min(0.92, 0.4 + spread * 1.2).toFixed(3);
                } else if (g.type === 'coal_steam') {
                    result.cf_sim = +Math.min(0.85, 0.3 + spread * 1.0).toFixed(3);
                } else {
                    result.cf_sim = +Math.min(0.3, 0.02 + spread * 0.5).toFixed(3);
                }
            } else {
                result.cf_sim = g.type === 'gas_ct' ? 0.02 : 0.0;
            }

            const dispatchHrs = Math.round(result.cf_sim * 8760);
            const energyRev = avgLMP * result.cf_sim;
            const profit = energyRev - mc * result.cf_sim;

            result.dispatch_hrs = dispatchHrs;
            result.energy_rev = +energyRev.toFixed(2);
            result.annual_profit_mw = +(profit * 8760).toFixed(0);  // $/MW-yr
            result.profit_mwh = +((energyRev - mc * result.cf_sim) / Math.max(result.cf_sim, 0.01)).toFixed(2);

            if (profit < -2) result.status = 'retiring';
            else if (profit < 3) result.status = 'marginal';
            else result.status = 'profitable';

        } else if (g.fuel === 'nuclear') {
            const energyRev = avgLMP * NUCLEAR_CF;
            const ptcRev = NUCLEAR_PTC;
            const totalRev = energyRev + ptcRev;
            result.mc = NUCLEAR_COST_MWH;
            result.cf_sim = NUCLEAR_CF;
            result.energy_rev = +energyRev.toFixed(2);
            result.ptc_rev = ptcRev;
            result.total_rev = +totalRev.toFixed(2);
            result.annual_profit_mw = +((totalRev - NUCLEAR_COST_MWH) * 8760 * NUCLEAR_CF).toFixed(0);
            result.status = totalRev > NUCLEAR_COST_MWH ? 'profitable' : 'at_risk';

        } else if (g.fuel === 'wind') {
            const cf = g.cf || 0.35;
            result.cf_sim = cf;
            result.energy_rev = +(avgLMP * cf).toFixed(2);
            result.annual_profit_mw = +(avgLMP * cf * 8760).toFixed(0);
            result.status = 'profitable';

        } else if (g.fuel === 'solar') {
            const cf = g.cf || 0.22;
            // Solar captures ~80% of avg LMP (production-weighted)
            const captureRate = 0.80;
            result.cf_sim = cf;
            result.energy_rev = +(avgLMP * captureRate * cf).toFixed(2);
            result.annual_profit_mw = +(avgLMP * captureRate * cf * 8760).toFixed(0);
            result.status = 'profitable';

        } else if (g.fuel === 'battery') {
            // Battery revenue from arbitrage
            const arbSpread = avgLMP * 0.3; // 30% of avg LMP as typical spread
            const cycles = 300; // per year
            const duration = 4; // hours
            result.cf_sim = 0;
            result.energy_rev = +(arbSpread * cycles * duration / 8760).toFixed(2);
            result.annual_profit_mw = +(arbSpread * cycles * duration).toFixed(0);
            result.status = 'profitable';

        } else {
            result.cf_sim = g.cf || 0;
            result.energy_rev = +(avgLMP * (g.cf || 0)).toFixed(2);
            result.status = 'operating';
        }

        return result;
    });

    // ── Aggregate by owner ──
    const ownerAgg = {};
    for (const g of results) {
        const owner = g.owner;
        if (!ownerAgg[owner]) {
            ownerAgg[owner] = {
                name: owner,
                total_mw: 0, fossil_mw: 0, nuclear_mw: 0, renewable_mw: 0, battery_mw: 0,
                total_revenue: 0, total_cost: 0, total_profit: 0,
                fossil_profit: 0, nuclear_profit: 0, renewable_profit: 0,
                units: 0, retiring_mw: 0, marginal_mw: 0,
                co2_tons: 0,
                by_type: {},
            };
        }
        const o = ownerAgg[owner];
        o.total_mw += g.mw;
        o.units += 1;

        const annualGen = (g.cf_sim || 0) * g.mw * 8760;
        const annualRev = (g.energy_rev || 0) * g.mw * 8760 / 1e6; // $M
        const annualProfit = (g.annual_profit_mw || 0) * g.mw / 1e6; // $M

        o.total_revenue += annualRev;

        if (g.fuel === 'coal' || g.fuel === 'gas' || g.fuel === 'oil') {
            o.fossil_mw += g.mw;
            o.fossil_profit += annualProfit;
            o.co2_tons += (g.co2_rate || 0) * annualGen;
            const annualCost = (g.mc || 0) * (g.cf_sim || 0) * g.mw * 8760 / 1e6;
            o.total_cost += annualCost;
        } else if (g.fuel === 'nuclear') {
            o.nuclear_mw += g.mw;
            o.nuclear_profit += annualProfit;
        } else if (g.fuel === 'wind' || g.fuel === 'solar' || g.fuel === 'hydro') {
            o.renewable_mw += g.mw;
            o.renewable_profit += annualProfit;
        } else if (g.fuel === 'battery') {
            o.battery_mw += g.mw;
        }

        if (g.status === 'retiring') o.retiring_mw += g.mw;
        if (g.status === 'marginal') o.marginal_mw += g.mw;

        // Aggregate by type
        if (!o.by_type[g.type]) {
            o.by_type[g.type] = { mw: 0, gen_gwh: 0, rev_m: 0, profit_m: 0 };
        }
        o.by_type[g.type].mw += g.mw;
        o.by_type[g.type].gen_gwh += annualGen / 1000;
        o.by_type[g.type].rev_m += annualRev;
        o.by_type[g.type].profit_m += annualProfit;
    }

    // Convert to sorted array
    const ownersList = Object.values(ownerAgg)
        .filter(o => o.total_mw > 50)
        .sort((a, b) => b.total_mw - a.total_mw)
        .map(o => ({
            ...o,
            total_revenue: +o.total_revenue.toFixed(1),
            total_cost: +o.total_cost.toFixed(1),
            total_profit: +(o.total_revenue - o.total_cost).toFixed(1),
            fossil_profit: +o.fossil_profit.toFixed(1),
            nuclear_profit: +o.nuclear_profit.toFixed(1),
            renewable_profit: +o.renewable_profit.toFixed(1),
            co2_tons: Math.round(o.co2_tons),
            total_mw: Math.round(o.total_mw),
            fossil_mw: Math.round(o.fossil_mw),
            nuclear_mw: Math.round(o.nuclear_mw),
            renewable_mw: Math.round(o.renewable_mw),
            retiring_mw: Math.round(o.retiring_mw),
            marginal_mw: Math.round(o.marginal_mw),
        }));

    // ── CCGT heat rate bin analysis ──
    const ccgtBins = {};
    for (const g of results) {
        if (g.hr_bin) {
            if (!ccgtBins[g.hr_bin]) {
                ccgtBins[g.hr_bin] = {
                    label: g.hr_bin_label,
                    count: 0, mw: 0, avg_mc: 0, avg_cf: 0, avg_profit: 0,
                    total_mc_mw: 0, total_cf_mw: 0, total_profit_mw: 0,
                    retiring_mw: 0, marginal_mw: 0, profitable_mw: 0,
                };
            }
            const b = ccgtBins[g.hr_bin];
            b.count += 1;
            b.mw += g.mw;
            b.total_mc_mw += (g.mc || 0) * g.mw;
            b.total_cf_mw += (g.cf_sim || 0) * g.mw;
            b.total_profit_mw += (g.annual_profit_mw || 0) * g.mw;
            if (g.status === 'retiring') b.retiring_mw += g.mw;
            else if (g.status === 'marginal') b.marginal_mw += g.mw;
            else b.profitable_mw += g.mw;
        }
    }
    // Compute averages
    for (const [tag, b] of Object.entries(ccgtBins)) {
        if (b.mw > 0) {
            b.avg_mc = +(b.total_mc_mw / b.mw).toFixed(2);
            b.avg_cf = +(b.total_cf_mw / b.mw).toFixed(3);
            b.avg_profit = +(b.total_profit_mw / b.mw).toFixed(0);
        }
        b.mw = Math.round(b.mw);
        b.retiring_mw = Math.round(b.retiring_mw);
        b.marginal_mw = Math.round(b.marginal_mw);
        b.profitable_mw = Math.round(b.profitable_mw);
        delete b.total_mc_mw;
        delete b.total_cf_mw;
        delete b.total_profit_mw;
    }

    // ── Fleet summary ──
    const fossilGens = results.filter(g => g.fuel === 'coal' || g.fuel === 'gas' || g.fuel === 'oil');
    const nuclearGens = results.filter(g => g.fuel === 'nuclear');
    const windGens = results.filter(g => g.fuel === 'wind');
    const solarGens = results.filter(g => g.fuel === 'solar');
    const batteryGens = results.filter(g => g.fuel === 'battery');

    const totalFossilMW = fossilGens.reduce((s, g) => s + g.mw, 0);
    const retiringMW = fossilGens.filter(g => g.status === 'retiring').reduce((s, g) => s + g.mw, 0);
    const marginalMW = fossilGens.filter(g => g.status === 'marginal').reduce((s, g) => s + g.mw, 0);

    const totalCO2 = fossilGens.reduce((s, g) => {
        return s + (g.co2_rate || 0) * (g.cf_sim || 0) * g.mw * 8760;
    }, 0);

    // CCS breakeven
    const avgCCGT_MC = fossilGens
        .filter(g => g.type === 'gas_ccgt')
        .reduce((s, g) => s + (g.mc || 0) * g.mw, 0) /
        Math.max(1, fossilGens.filter(g => g.type === 'gas_ccgt').reduce((s, g) => s + g.mw, 0));
    const ccsLCOE = NEW_BUILD_LCOE[snapshotYear]?.ccs || 72;
    const ccsBreakeven = Math.max(0, (ccsLCOE - avgCCGT_MC) / 0.33); // 90% capture = 0.37-0.037

    // Nuclear revenue stack
    const nucRevEnergy = avgLMP * NUCLEAR_CF;
    const nucRevPTC = NUCLEAR_PTC;
    const nucTotalRev = nucRevEnergy + nucRevPTC;

    return {
        meta: {
            fuel_level: fuelLevel,
            carbon_price: carbonPrice,
            snapshot_year: snapshotYear,
            fuel_prices: fuelPrices,
        },
        fleet_summary: {
            total_generators: results.length,
            total_mw: Math.round(results.reduce((s, g) => s + g.mw, 0)),
            fossil_mw: Math.round(totalFossilMW),
            nuclear_mw: Math.round(nuclearGens.reduce((s, g) => s + g.mw, 0)),
            wind_mw: Math.round(windGens.reduce((s, g) => s + g.mw, 0)),
            solar_mw: Math.round(solarGens.reduce((s, g) => s + g.mw, 0)),
            battery_mw: Math.round(batteryGens.reduce((s, g) => s + g.mw, 0)),
            avg_lmp: +avgLMP.toFixed(2),
            total_co2_mt: +(totalCO2 / 1e6).toFixed(1),
            retiring_mw: Math.round(retiringMW),
            marginal_mw: Math.round(marginalMW),
        },
        nuclear_revenue: {
            energy_mwh: +nucRevEnergy.toFixed(2),
            ptc_mwh: nucRevPTC,
            total_mwh: +nucTotalRev.toFixed(2),
            status: nucTotalRev > NUCLEAR_COST_MWH ? 'profitable' : 'at_risk',
        },
        ccs_breakeven_carbon: +ccsBreakeven.toFixed(0),
        ccgt_bins: ccgtBins,
        owners: ownersList.slice(0, 25),
        generators: results,
        new_build_lcoe: NEW_BUILD_LCOE[snapshotYear] || NEW_BUILD_LCOE[2025],
    };
}


/**
 * Run carbon price sweep for a specific owner to build trajectory.
 */
function runOwnerTrajectory(ownerName, fuelLevel, years) {
    years = years || YEAR_SNAPSHOTS;
    const carbonPrices = [0, 10, 20, 35, 50, 75, 100, 150, 200];

    const trajectories = [];
    for (const year of years) {
        for (const cp of carbonPrices) {
            const sim = runFleetSimulation(fuelLevel, cp, year);
            if (!sim) continue;

            const owner = sim.owners.find(o => o.name === ownerName);
            if (!owner) continue;

            trajectories.push({
                year: year,
                carbon_price: cp,
                total_mw: owner.total_mw,
                fossil_mw: owner.fossil_mw,
                nuclear_mw: owner.nuclear_mw,
                renewable_mw: owner.renewable_mw,
                total_revenue: owner.total_revenue,
                fossil_profit: owner.fossil_profit,
                nuclear_profit: owner.nuclear_profit,
                renewable_profit: owner.renewable_profit,
                retiring_mw: owner.retiring_mw,
                co2_tons: owner.co2_tons,
            });
        }
    }

    return trajectories;
}


/**
 * Run CCGT stranding analysis: how do heat rate bins fare across carbon prices?
 */
function runCCGTStrandingAnalysis(fuelLevel, snapshotYear) {
    const carbonPrices = [0, 10, 20, 35, 50, 75, 100, 150, 200];
    const binData = {};

    for (const cp of carbonPrices) {
        const sim = runFleetSimulation(fuelLevel, cp, snapshotYear);
        if (!sim) continue;

        for (const [tag, bin] of Object.entries(sim.ccgt_bins)) {
            if (!binData[tag]) {
                binData[tag] = { label: bin.label, points: [] };
            }
            binData[tag].points.push({
                carbon_price: cp,
                avg_mc: bin.avg_mc,
                avg_cf: bin.avg_cf,
                avg_profit_mw_yr: bin.avg_profit,
                profitable_mw: bin.profitable_mw,
                marginal_mw: bin.marginal_mw,
                retiring_mw: bin.retiring_mw,
            });
        }
    }

    return binData;
}


/**
 * Map EIA entity names to curated IPP company IDs.
 * EIA names don't always match curated names exactly.
 */
const EIA_TO_IPP_MAP = {
    'Luminant Generation Company LLC': 'vistra',
    'Vistra Operations Company LLC': 'vistra',
    'TXU Energy Retail Company LLC': 'vistra',
    'Constellation Power, Inc': 'constellation',
    'Calpine Corporation': 'constellation',  // Post-merger Jan 2026
    'NRG Texas Power LLC': 'nrg',
    'NRG Texas LP': 'nrg',
    'GenOn Energy': 'nrg',
    'NextEra Energy Resources': 'nextera',
    'FPL Group': 'nextera',
    'STP Nuclear Operating Co': 'constellation',  // 44% Constellation share
};


/**
 * Build comprehensive company portfolio combining EIA fleet data + curated IPP data.
 * Returns enriched owner records with national portfolio context.
 */
function buildCompanyPortfolios(sim) {
    if (!sim || !window.IPP_DATA) return sim.owners;

    const enriched = sim.owners.map(owner => {
        // Try to match to curated IPP
        const ippId = EIA_TO_IPP_MAP[owner.name];
        const ippCompany = ippId ? IPP_DATA.companies.find(c => c.id === ippId) : null;

        if (ippCompany) {
            return {
                ...owner,
                ipp_id: ippId,
                short_name: ippCompany.shortName,
                target: ippCompany.target,
                national: ippCompany.national_portfolio,
                curated_ercot: ippCompany.ercot_plants,
                co2_company_total_mt: ippCompany.co2_2024_mt,
                gen_company_total_twh: ippCompany.gen_twh,
                cap_company_total_gw: ippCompany.cap_gw,
            };
        }
        return owner;
    });

    return enriched;
}


/**
 * Compute full portfolio revenue trajectory for a company across years and carbon prices.
 * Includes national nuclear + RE revenue estimates (not just ERCOT).
 */
function runCompanyFullTrajectory(ippId, fuelLevel) {
    if (!window.IPP_DATA) return null;
    const company = IPP_DATA.companies.find(c => c.id === ippId);
    if (!company) return null;

    const years = YEAR_SNAPSHOTS;
    const carbonPrices = [0, 10, 20, 35, 50, 75, 100, 150, 200];
    const points = [];

    for (const year of years) {
        for (const cp of carbonPrices) {
            const sim = runFleetSimulation(fuelLevel, cp, year);
            if (!sim) continue;

            const avgLMP = sim.fleet_summary.avg_lmp;

            // Find this company's ERCOT fleet in the simulation
            const eiaNames = Object.entries(EIA_TO_IPP_MAP)
                .filter(([_, id]) => id === ippId)
                .map(([name, _]) => name);

            const companyGens = sim.generators.filter(g => eiaNames.includes(g.owner));

            // ERCOT revenue
            let ercotFossilRev = 0, ercotNucRev = 0, ercotRERev = 0, ercotBattRev = 0;
            let ercotFossilMW = 0, ercotNucMW = 0, ercotREMW = 0;
            let ercotCO2 = 0, ercotRetiringMW = 0;

            for (const g of companyGens) {
                const annRev = (g.annual_profit_mw || 0) * g.mw / 1e6;
                if (g.fuel === 'coal' || g.fuel === 'gas' || g.fuel === 'oil') {
                    ercotFossilRev += annRev;
                    ercotFossilMW += g.mw;
                    ercotCO2 += (g.co2_rate || 0) * (g.cf_sim || 0) * g.mw * 8760;
                    if (g.status === 'retiring' || g.status === 'marginal') ercotRetiringMW += g.mw;
                } else if (g.fuel === 'nuclear') {
                    ercotNucRev += annRev;
                    ercotNucMW += g.mw;
                } else if (g.fuel === 'wind' || g.fuel === 'solar' || g.fuel === 'hydro') {
                    ercotRERev += annRev;
                    ercotREMW += g.mw;
                } else if (g.fuel === 'battery') {
                    ercotBattRev += annRev;
                }
            }

            // National portfolio (non-ERCOT) - estimate from curated data
            const nat = company.national_portfolio;
            const natNucMW = Math.max(0, nat.nuclear_mw - ercotNucMW);
            const natGasMW = Math.max(0, nat.gas_ccgt_mw + nat.gas_peaker_mw - ercotFossilMW);

            // Rough PJM/other-ISO nuclear revenue (higher than ERCOT due to capacity markets)
            const otherISOLMP = avgLMP * 1.1 + 5; // PJM/NYISO tend higher
            const capMarketRev = 120 / 8.76; // ~$13.7/MWh from PJM capacity
            const natNucRev = natNucMW > 0
                ? natNucMW * (otherISOLMP * NUCLEAR_CF + NUCLEAR_PTC + capMarketRev - NUCLEAR_COST_MWH) * 8760 / 1e6
                : 0;

            // National renewable revenue
            const natWindMW = nat.wind_mw || 0;
            const natSolarMW = nat.solar_mw || 0;
            const natGeoMW = nat.geothermal_mw || 0;
            const natRERev = (natWindMW * avgLMP * 0.35 + natSolarMW * avgLMP * 0.22 * 0.8 + natGeoMW * avgLMP * 0.90) * 8760 / 1e6;

            // National gas revenue
            const fp = FUEL_PRICES[fuelLevel];
            const natGasMC = fp.gas * 7.0 + 3.50 + cp * 0.0531 * 7.0;
            const natGasLMP = otherISOLMP;
            const natGasCF = natGasMC < natGasLMP ? Math.min(0.75, 0.4 + (natGasLMP - natGasMC) / natGasLMP) : 0.05;
            const natGasRev = natGasMW > 0
                ? natGasMW * (natGasLMP * natGasCF - natGasMC * natGasCF) * 8760 / 1e6
                : 0;

            points.push({
                year,
                carbon_price: cp,
                // ERCOT
                ercot_fossil_rev: +ercotFossilRev.toFixed(1),
                ercot_nuclear_rev: +ercotNucRev.toFixed(1),
                ercot_re_rev: +ercotRERev.toFixed(1),
                ercot_battery_rev: +ercotBattRev.toFixed(1),
                ercot_fossil_mw: Math.round(ercotFossilMW),
                ercot_retiring_mw: Math.round(ercotRetiringMW),
                ercot_co2_mt: +(ercotCO2 / 1e6).toFixed(2),
                // National (non-ERCOT)
                national_nuclear_rev: +natNucRev.toFixed(1),
                national_gas_rev: +natGasRev.toFixed(1),
                national_re_rev: +natRERev.toFixed(1),
                // Totals
                total_rev: +(ercotFossilRev + ercotNucRev + ercotRERev + ercotBattRev + natNucRev + natGasRev + natRERev).toFixed(1),
                avg_lmp: avgLMP,
            });
        }
    }

    return {
        company: company.shortName,
        ipp_id: ippId,
        target: company.target,
        national_portfolio: company.national_portfolio,
        points,
    };
}


/**
 * Run competitor comparison: compute portfolio metrics for all ERCOT IPPs at one scenario.
 */
function runCompetitorComparison(fuelLevel, carbonPrice, snapshotYear) {
    if (!window.IPP_DATA) return [];

    const sim = runFleetSimulation(fuelLevel, carbonPrice, snapshotYear);
    if (!sim) return [];

    const results = [];
    for (const company of IPP_DATA.companies) {
        const eiaNames = Object.entries(EIA_TO_IPP_MAP)
            .filter(([_, id]) => id === company.id)
            .map(([name, _]) => name);

        const gens = sim.generators.filter(g => eiaNames.includes(g.owner));
        if (!gens.length) continue;

        let fossilRev = 0, nucRev = 0, reRev = 0, totalMW = 0;
        let fossilMW = 0, nucMW = 0, reMW = 0, battMW = 0;
        let retiringMW = 0, co2 = 0;

        for (const g of gens) {
            totalMW += g.mw;
            const profit = (g.annual_profit_mw || 0) * g.mw / 1e6;
            if (g.fuel === 'coal' || g.fuel === 'gas' || g.fuel === 'oil') {
                fossilRev += profit;
                fossilMW += g.mw;
                co2 += (g.co2_rate || 0) * (g.cf_sim || 0) * g.mw * 8760;
                if (g.status === 'retiring' || g.status === 'marginal') retiringMW += g.mw;
            } else if (g.fuel === 'nuclear') {
                nucRev += profit;
                nucMW += g.mw;
            } else if (g.fuel === 'wind' || g.fuel === 'solar' || g.fuel === 'hydro') {
                reRev += profit;
                reMW += g.mw;
            } else if (g.fuel === 'battery') {
                battMW += g.mw;
            }
        }

        const nat = company.national_portfolio;
        results.push({
            id: company.id,
            name: company.shortName,
            target: company.target,
            // ERCOT
            ercot_total_mw: Math.round(totalMW),
            ercot_fossil_mw: Math.round(fossilMW),
            ercot_nuclear_mw: Math.round(nucMW),
            ercot_re_mw: Math.round(reMW),
            ercot_battery_mw: Math.round(battMW),
            ercot_fossil_rev: +fossilRev.toFixed(1),
            ercot_nuclear_rev: +nucRev.toFixed(1),
            ercot_re_rev: +reRev.toFixed(1),
            ercot_retiring_mw: Math.round(retiringMW),
            ercot_co2_mt: +(co2 / 1e6).toFixed(2),
            // National
            national_nuclear_mw: nat.nuclear_mw,
            national_total_gen_twh: nat.total_gen_twh,
            national_total_co2_mt: nat.total_co2_mt,
            // Exposure metrics
            fossil_pct: totalMW > 0 ? +((fossilMW / totalMW) * 100).toFixed(1) : 0,
            retiring_pct: totalMW > 0 ? +((retiringMW / totalMW) * 100).toFixed(1) : 0,
            clean_pct: totalMW > 0 ? +(((nucMW + reMW + battMW) / totalMW) * 100).toFixed(1) : 0,
        });
    }

    return results.sort((a, b) => b.ercot_total_mw - a.ercot_total_mw);
}
