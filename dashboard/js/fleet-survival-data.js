/**
 * Fleet Survival Dashboard - Data Module
 * ============================================================================
 * Contains: company fleet data by ISO/fuel, fossil dispatch decline curves
 * from the Track 1 baseline pipeline, SBTi timeline mapping, and 2024
 * CO2 emission estimates (MJ Bradley / ERM methodology).
 *
 * Sources:
 *   - Company data: EIA Form 860/923, eGRID, company sustainability reports,
 *     MJ Bradley/ERM Benchmarking Air Emissions (21st ed., 2025)
 *   - Dispatch decline: hourly-cfe-optimizer Step 4/PP4 merit-order retirement
 *   - SBTi timeline: shared-data.js THRESHOLD_TARGET_YEARS
 */

// ============================================================================
// SBTi TIMELINE (mirrors shared-data.js)
// ============================================================================
const FLEET_THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.99];
const FLEET_THRESHOLD_YEARS = {
    50: 2030, 55: 2031, 60: 2033, 65: 2034, 70: 2035,
    75: 2036, 80: 2037, 85: 2038, 87.5: 2039, 90: 2040,
    92.5: 2043, 95: 2045, 97.5: 2048, 99: 2049, 99.99: 2050,
};

// ============================================================================
// COMPANY FLEET DATA
// ============================================================================
// Each company has:
//   - name, shortName, co2_2024_mt (total 2024 CO2 in million metric tons)
//   - intensity_kg_per_mwh (2024 fleet-weighted average)
//   - plants: array of { name, iso, fuel, subcategory, capacity_mw, generation_twh,
//                         co2_mt (annual CO2 in million metric tons), heat_rate_btu }
//
// Fuel categories: coal, gas_ccgt, gas_peaker, oil, nuclear, solar, wind, battery, hydro, geothermal
// Subcategory for gas: 'ccgt' (combined cycle) or 'peaker' (combustion turbine / simple cycle)

const FLEET_COMPANIES = [
    {
        name: "Vistra Energy",
        shortName: "Vistra",
        type: "Merchant/IPP",
        co2_2024_mt: 68,
        intensity_kg_per_mwh: 485,
        total_generation_twh: 154,
        total_capacity_gw: 44,
        target: "Net-zero by 2050",
        pipeline_twh: 4.5, // Solar/battery in ERCOT + Comanche Peak nuclear uprate
        plants: [
            // ERCOT - Coal (Martin Lake + Coleto Creek; retiring by 2027)
            { name: "Martin Lake Coal (TX)", iso: "ERCOT", fuel: "coal", capacity_mw: 2250, generation_twh: 11.0, co2_mt: 11.6, heat_rate_btu: 10200 },
            { name: "Coleto Creek Coal (TX)", iso: "ERCOT", fuel: "coal", capacity_mw: 630, generation_twh: 3.1, co2_mt: 3.3, heat_rate_btu: 10400 },
            // ERCOT - Gas CCGT
            { name: "Odessa-Ector CCGT", iso: "ERCOT", fuel: "gas_ccgt", capacity_mw: 1054, generation_twh: 5.5, co2_mt: 2.2 },
            { name: "Midlothian CCGT", iso: "ERCOT", fuel: "gas_ccgt", capacity_mw: 1600, generation_twh: 8.4, co2_mt: 3.3 },
            { name: "Forney CCGT", iso: "ERCOT", fuel: "gas_ccgt", capacity_mw: 1800, generation_twh: 9.4, co2_mt: 3.8 },
            // ERCOT - Gas Peaker (Hays, Stryker Creek, Graham, Morgan Creek, Permian, Trinidad, etc.)
            { name: "ERCOT CT/Peaker Fleet (8 plants)", iso: "ERCOT", fuel: "gas_peaker", capacity_mw: 4600, generation_twh: 4.0, co2_mt: 2.6 },
            // ERCOT - Nuclear
            { name: "Comanche Peak Nuclear", iso: "ERCOT", fuel: "nuclear", capacity_mw: 2400, generation_twh: 20.5, co2_mt: 0 },
            // ERCOT - Solar + Battery
            { name: "ERCOT Solar Portfolio", iso: "ERCOT", fuel: "solar", capacity_mw: 358, generation_twh: 0.8, co2_mt: 0 },
            { name: "DeCordova BESS", iso: "ERCOT", fuel: "battery", capacity_mw: 260, generation_twh: 0, co2_mt: 0 },
            // PJM - Nuclear (Perry, Davis-Besse, Beaver Valley)
            { name: "Perry Nuclear (OH)", iso: "PJM", fuel: "nuclear", capacity_mw: 1268, generation_twh: 10.7, co2_mt: 0 },
            { name: "Davis-Besse Nuclear (OH)", iso: "PJM", fuel: "nuclear", capacity_mw: 908, generation_twh: 7.6, co2_mt: 0 },
            { name: "Beaver Valley Nuclear (PA)", iso: "PJM", fuel: "nuclear", capacity_mw: 1872, generation_twh: 15.8, co2_mt: 0 },
            // PJM/MISO - Coal (Kincaid, Baldwin, Newton; all retiring 2027)
            { name: "Kincaid Coal (IL)", iso: "PJM", fuel: "coal", capacity_mw: 1319, generation_twh: 4.0, co2_mt: 4.0, heat_rate_btu: 10800 },
            { name: "Baldwin Coal (IL)", iso: "PJM", fuel: "coal", capacity_mw: 1185, generation_twh: 3.6, co2_mt: 3.6, heat_rate_btu: 10600 },
            { name: "Newton Coal (IL)", iso: "PJM", fuel: "coal", capacity_mw: 615, generation_twh: 1.9, co2_mt: 1.9, heat_rate_btu: 10700 },
            // PJM - Gas CCGT (ex-Dynegy fleet)
            { name: "PJM Gas CCGT Fleet (ex-Dynegy)", iso: "PJM", fuel: "gas_ccgt", capacity_mw: 5200, generation_twh: 27.3, co2_mt: 10.9 },
            // PJM - Gas Peaker
            { name: "PJM CT/Peaker Fleet", iso: "PJM", fuel: "gas_peaker", capacity_mw: 5000, generation_twh: 4.4, co2_mt: 2.8 },
            // CAISO - Gas + Battery (Moss Landing)
            { name: "Moss Landing CCGT (CA)", iso: "CAISO", fuel: "gas_ccgt", capacity_mw: 1200, generation_twh: 6.3, co2_mt: 2.5 },
            // NEISO - Gas CCGT (Cogentrix pending: Newington, Bridgeport, Tiverton, Rumford)
            { name: "Newington CCGT (NH)", iso: "NEISO", fuel: "gas_ccgt", capacity_mw: 624, generation_twh: 3.3, co2_mt: 1.3 },
            { name: "Bridgeport CCGT (CT)", iso: "NEISO", fuel: "gas_ccgt", capacity_mw: 558, generation_twh: 2.9, co2_mt: 1.2 },
            { name: "Tiverton/Rumford CCGT", iso: "NEISO", fuel: "gas_ccgt", capacity_mw: 568, generation_twh: 3.0, co2_mt: 1.2 },
        ]
    },
    {
        name: "Constellation Energy",
        shortName: "Constellation",
        type: "Merchant/IPP",
        co2_2024_mt: 55,
        intensity_kg_per_mwh: 177,
        total_generation_twh: 310,
        total_capacity_gw: 55,
        target: "100% carbon-free by 2040",
        pipeline_twh: 15.3, // ~8 TWh nuclear uprates (NRC-approved) + ~6.5 TWh Crane restart + renewables/geo
        plants: [
            // PJM - Nuclear (Limerick, Peach Bottom 50%, Calvert Cliffs, Crane/TMI restart, Salem 43%)
            { name: "Limerick Nuclear (PA)", iso: "PJM", fuel: "nuclear", capacity_mw: 2317, generation_twh: 19.5, co2_mt: 0 },
            { name: "Peach Bottom Nuclear (PA, 50%)", iso: "PJM", fuel: "nuclear", capacity_mw: 1385, generation_twh: 11.7, co2_mt: 0 },
            { name: "Calvert Cliffs Nuclear (MD)", iso: "PJM", fuel: "nuclear", capacity_mw: 1790, generation_twh: 15.1, co2_mt: 0 },
            { name: "Crane Clean Energy Center (PA)", iso: "PJM", fuel: "nuclear", capacity_mw: 835, generation_twh: 7.0, co2_mt: 0 },
            { name: "Salem Nuclear (NJ, 43%)", iso: "PJM", fuel: "nuclear", capacity_mw: 987, generation_twh: 8.3, co2_mt: 0 },
            // PJM - Gas (Calpine: Zion CT, Cumberland CT, Sherman Ave CT; ~3.5 GW being divested)
            { name: "PJM Gas Peaker Fleet", iso: "PJM", fuel: "gas_peaker", capacity_mw: 786, generation_twh: 0.7, co2_mt: 0.4 },
            // PJM - Hydro
            { name: "Conowingo Hydro (MD)", iso: "PJM", fuel: "hydro", capacity_mw: 572, generation_twh: 1.9, co2_mt: 0 },
            // NYISO - Nuclear (Nine Mile Point, FitzPatrick, Ginna)
            { name: "Nine Mile Point Nuclear (NY)", iso: "NYISO", fuel: "nuclear", capacity_mw: 1907, generation_twh: 16.1, co2_mt: 0 },
            { name: "FitzPatrick Nuclear (NY)", iso: "NYISO", fuel: "nuclear", capacity_mw: 842, generation_twh: 7.1, co2_mt: 0 },
            { name: "Ginna Nuclear (NY)", iso: "NYISO", fuel: "nuclear", capacity_mw: 576, generation_twh: 4.9, co2_mt: 0 },
            // NYISO - Gas (Calpine: Bethpage CT, Kennedy cogen)
            { name: "NYISO Calpine Gas Fleet", iso: "NYISO", fuel: "gas_peaker", capacity_mw: 225, generation_twh: 0.5, co2_mt: 0.2 },
            // NEISO - Gas CCGT (Calpine: Fore River, Granite Ridge, Westbrook)
            { name: "Fore River CCGT (MA)", iso: "NEISO", fuel: "gas_ccgt", capacity_mw: 750, generation_twh: 3.9, co2_mt: 1.5 },
            { name: "Granite Ridge CCGT (NH)", iso: "NEISO", fuel: "gas_ccgt", capacity_mw: 745, generation_twh: 3.9, co2_mt: 1.5 },
            { name: "Westbrook CCGT (ME)", iso: "NEISO", fuel: "gas_ccgt", capacity_mw: 552, generation_twh: 2.9, co2_mt: 1.1 },
            // ERCOT - Nuclear (South Texas Project 44%)
            { name: "South Texas Project (44%)", iso: "ERCOT", fuel: "nuclear", capacity_mw: 1164, generation_twh: 9.8, co2_mt: 0 },
            // ERCOT - Gas CCGT (Calpine: Deer Park, Guadalupe, Baytown, Channel, Freestone, Bosque, etc.)
            { name: "Deer Park Energy Center", iso: "ERCOT", fuel: "gas_ccgt", capacity_mw: 1204, generation_twh: 6.3, co2_mt: 2.5 },
            { name: "Guadalupe Energy Center", iso: "ERCOT", fuel: "gas_ccgt", capacity_mw: 1009, generation_twh: 5.3, co2_mt: 2.1 },
            { name: "Baytown/Channel/Freestone CCGT", iso: "ERCOT", fuel: "gas_ccgt", capacity_mw: 2483, generation_twh: 13.0, co2_mt: 5.2 },
            { name: "Bosque/Magic Valley/Pasadena CCGT", iso: "ERCOT", fuel: "gas_ccgt", capacity_mw: 2083, generation_twh: 10.9, co2_mt: 4.4 },
            { name: "Quail Run/Corpus/Lost Pines CCGT", iso: "ERCOT", fuel: "gas_ccgt", capacity_mw: 1570, generation_twh: 8.2, co2_mt: 3.3 },
            // ERCOT - Gas Peaker (Hidalgo + others)
            { name: "ERCOT Calpine Peaker Fleet", iso: "ERCOT", fuel: "gas_peaker", capacity_mw: 1500, generation_twh: 1.3, co2_mt: 0.8 },
            // ERCOT - Constellation legacy gas
            { name: "Colorado Bend CCGT (TX)", iso: "ERCOT", fuel: "gas_ccgt", capacity_mw: 550, generation_twh: 2.9, co2_mt: 1.2 },
            // CAISO - Gas CCGT (Calpine: Delta, Marsh Landing, Los Medanos, Russell City, Metcalf, Gateway, Pastoria, Sutter)
            { name: "Delta/Los Medanos CCGT", iso: "CAISO", fuel: "gas_ccgt", capacity_mw: 1643, generation_twh: 8.6, co2_mt: 3.4 },
            { name: "Russell City/Gateway CCGT", iso: "CAISO", fuel: "gas_ccgt", capacity_mw: 1239, generation_twh: 6.5, co2_mt: 2.6 },
            { name: "Pastoria/Metcalf/Sutter CCGT", iso: "CAISO", fuel: "gas_ccgt", capacity_mw: 1754, generation_twh: 9.2, co2_mt: 3.7 },
            // CAISO - Gas Peaker (Marsh Landing, Los Esteros, Gilroy)
            { name: "CAISO Calpine Peaker Fleet", iso: "CAISO", fuel: "gas_peaker", capacity_mw: 1318, generation_twh: 1.2, co2_mt: 0.7 },
            // CAISO - Geothermal (The Geysers, 13 plants)
            { name: "The Geysers Geothermal (CA)", iso: "CAISO", fuel: "geothermal", capacity_mw: 725, generation_twh: 5.3, co2_mt: 0.3 },
            // CAISO - Battery (Nova BESS)
            { name: "Nova BESS Portfolio (CA)", iso: "CAISO", fuel: "battery", capacity_mw: 798, generation_twh: 0, co2_mt: 0 },
            // Wind/Solar across ISOs
            { name: "Wind Portfolio (10 states)", iso: "PJM", fuel: "wind", capacity_mw: 1400, generation_twh: 4.6, co2_mt: 0 },
            { name: "Solar Portfolio", iso: "CAISO", fuel: "solar", capacity_mw: 360, generation_twh: 0.8, co2_mt: 0 },
        ]
    },
    {
        name: "NRG Energy",
        shortName: "NRG",
        type: "Merchant/IPP",
        co2_2024_mt: 42,
        intensity_kg_per_mwh: 520,
        total_generation_twh: 80,
        total_capacity_gw: 25,
        target: "50% CO2 reduction by 2025 (vs 2014); net-zero by 2050",
        pipeline_twh: 1.5, // Minimal clean pipeline, primarily thermal/retail focus
        plants: [
            // ERCOT - Coal
            { name: "W.A. Parish Coal (TX)", iso: "ERCOT", fuel: "coal", capacity_mw: 2697, generation_twh: 13.0, co2_mt: 13.7, heat_rate_btu: 10500 },
            { name: "Limestone Coal (TX)", iso: "ERCOT", fuel: "coal", capacity_mw: 1629, generation_twh: 7.8, co2_mt: 8.2, heat_rate_btu: 10400 },
            // ERCOT - Gas CCGT (Parish gas units, Cedar Bayou, LS Power TX plants)
            { name: "W.A. Parish Gas CCGT", iso: "ERCOT", fuel: "gas_ccgt", capacity_mw: 1086, generation_twh: 5.7, co2_mt: 2.2 },
            { name: "Cedar Bayou CCGT (TX)", iso: "ERCOT", fuel: "gas_ccgt", capacity_mw: 1492, generation_twh: 7.8, co2_mt: 3.1 },
            { name: "LS Power TX Gas Plants (3)", iso: "ERCOT", fuel: "gas_ccgt", capacity_mw: 2060, generation_twh: 10.8, co2_mt: 4.3 },
            // ERCOT - Gas Peaker (T.H. Wharton, Bertron, Greens Bayou, San Jacinto, Rockland)
            { name: "ERCOT CT/Peaker Fleet (6 plants)", iso: "ERCOT", fuel: "gas_peaker", capacity_mw: 3489, generation_twh: 3.1, co2_mt: 2.0 },
            // PJM - Gas CCGT (LS Power: Doswell, Hunterstown, Ironwood, Springdale + others)
            { name: "Doswell Energy Center (VA)", iso: "PJM", fuel: "gas_ccgt", capacity_mw: 1313, generation_twh: 6.9, co2_mt: 2.8 },
            { name: "Hunterstown CCGT (PA)", iso: "PJM", fuel: "gas_ccgt", capacity_mw: 810, generation_twh: 4.2, co2_mt: 1.7 },
            { name: "Ironwood CCGT (PA)", iso: "PJM", fuel: "gas_ccgt", capacity_mw: 778, generation_twh: 4.1, co2_mt: 1.6 },
            { name: "Springdale/Other PJM CCGT", iso: "PJM", fuel: "gas_ccgt", capacity_mw: 2623, generation_twh: 13.8, co2_mt: 5.5 },
            // PJM - Gas Peaker
            { name: "PJM CT/Peaker Fleet", iso: "PJM", fuel: "gas_peaker", capacity_mw: 2000, generation_twh: 1.8, co2_mt: 1.1 },
            // NYISO - Gas (Ravenswood, Arthur Kill)
            { name: "Ravenswood Gas/Oil (NY)", iso: "NYISO", fuel: "gas_ccgt", capacity_mw: 1947, generation_twh: 8.5, co2_mt: 3.5 },
            { name: "Arthur Kill Gas (NY)", iso: "NYISO", fuel: "gas_peaker", capacity_mw: 866, generation_twh: 0.8, co2_mt: 0.5 },
            // NEISO - Gas (LS Power NE plants)
            { name: "NE Gas Plants (2)", iso: "NEISO", fuel: "gas_ccgt", capacity_mw: 940, generation_twh: 4.9, co2_mt: 1.9 },
        ]
    },
    {
        name: "Talen Energy",
        shortName: "Talen",
        type: "Merchant/IPP",
        co2_2024_mt: 14,
        intensity_kg_per_mwh: 340,
        total_generation_twh: 42,
        total_capacity_gw: 10.7,
        target: "No formal net-zero target; exploring nuclear expansion (Cumulus data center)",
        pipeline_twh: 1.0, // Susquehanna nuclear uprate potential only; data center colocation focus
        plants: [
            // PJM - Nuclear (Susquehanna, 90% owned = 2,228 MW)
            { name: "Susquehanna Nuclear (PA, 90%)", iso: "PJM", fuel: "nuclear", capacity_mw: 2228, generation_twh: 18.8, co2_mt: 0 },
            // PJM - Gas CCGT (Lower Mt. Bethel)
            { name: "Lower Mt. Bethel CCGT (PA)", iso: "PJM", fuel: "gas_ccgt", capacity_mw: 610, generation_twh: 3.2, co2_mt: 1.3 },
            // PJM - Gas Peaker (Martins Creek CT, Camden)
            { name: "Martins Creek CT (PA)", iso: "PJM", fuel: "gas_peaker", capacity_mw: 1710, generation_twh: 1.5, co2_mt: 1.0 },
            { name: "Camden CT (NJ)", iso: "PJM", fuel: "gas_peaker", capacity_mw: 145, generation_twh: 0.1, co2_mt: 0.1 },
            // PJM - Coal (Brandon Shores RMR through 2029; Conemaugh 22%, Keystone 12%)
            { name: "Brandon Shores Coal (MD, RMR)", iso: "PJM", fuel: "coal", capacity_mw: 1295, generation_twh: 3.9, co2_mt: 3.9, heat_rate_btu: 10600 },
            { name: "Brunner Island (PA, coal ceasing 2028)", iso: "PJM", fuel: "coal", capacity_mw: 1419, generation_twh: 4.3, co2_mt: 4.3, heat_rate_btu: 10500 },
            { name: "Montour (PA, coal ceased 2025/gas)", iso: "PJM", fuel: "gas_ccgt", capacity_mw: 1508, generation_twh: 7.9, co2_mt: 3.2 },
            { name: "Conemaugh (PA, 22%)", iso: "PJM", fuel: "coal", capacity_mw: 392, generation_twh: 1.2, co2_mt: 1.2, heat_rate_btu: 10700 },
            { name: "Keystone (PA, 12%)", iso: "PJM", fuel: "coal", capacity_mw: 214, generation_twh: 0.6, co2_mt: 0.6, heat_rate_btu: 10800 },
            // PJM - Oil/Gas (H.A. Wagner, RMR through 2029)
            { name: "H.A. Wagner (MD, oil/gas)", iso: "PJM", fuel: "gas_peaker", capacity_mw: 838, generation_twh: 1.0, co2_mt: 0.7 },
        ]
    },
    {
        name: "PSEG Power",
        shortName: "PSEG",
        type: "Merchant/IPP",
        co2_2024_mt: 1,
        intensity_kg_per_mwh: 22,
        total_generation_twh: 40,
        total_capacity_gw: 3.8,
        target: "Net-zero operations by 2045",
        pipeline_twh: 4.8, // Nuclear uprates (Hope Creek/Salem) + offshore wind + solar
        plants: [
            // PJM - Nuclear (Hope Creek 100%, Salem 57%, Peach Bottom 50%)
            { name: "Hope Creek Nuclear (NJ)", iso: "PJM", fuel: "nuclear", capacity_mw: 1172, generation_twh: 9.9, co2_mt: 0 },
            { name: "Salem Nuclear (NJ, 57%)", iso: "PJM", fuel: "nuclear", capacity_mw: 1308, generation_twh: 11.0, co2_mt: 0 },
            { name: "Peach Bottom Nuclear (PA, 50%)", iso: "PJM", fuel: "nuclear", capacity_mw: 1275, generation_twh: 10.7, co2_mt: 0 },
        ]
    },
    {
        name: "NextEra Energy Resources",
        shortName: "NextEra",
        type: "Merchant/IPP",
        co2_2024_mt: 18,
        intensity_kg_per_mwh: 180,
        total_generation_twh: 100,
        total_capacity_gw: 33,
        target: "Real Zero emissions by 2045",
        pipeline_twh: 50.0, // Largest US clean pipeline: ~25 GW backlog (solar/wind/battery/offshore/SMR)
        plants: [
            // PJM - Nuclear
            { name: "Point Beach Nuclear (WI)", iso: "PJM", fuel: "nuclear", capacity_mw: 1200, generation_twh: 10.1, co2_mt: 0 },
            // ERCOT - Wind + Solar
            { name: "ERCOT Wind Portfolio", iso: "ERCOT", fuel: "wind", capacity_mw: 8000, generation_twh: 25.0, co2_mt: 0 },
            { name: "ERCOT Solar Portfolio", iso: "ERCOT", fuel: "solar", capacity_mw: 3500, generation_twh: 7.7, co2_mt: 0 },
            { name: "ERCOT Battery Storage", iso: "ERCOT", fuel: "battery", capacity_mw: 2000, generation_twh: 0, co2_mt: 0 },
            // ERCOT - Gas
            { name: "ERCOT Gas CCGT Fleet", iso: "ERCOT", fuel: "gas_ccgt", capacity_mw: 3200, generation_twh: 16.8, co2_mt: 6.7 },
            // PJM - Wind/Solar
            { name: "PJM Wind Portfolio", iso: "PJM", fuel: "wind", capacity_mw: 2000, generation_twh: 6.3, co2_mt: 0 },
            { name: "PJM Solar Portfolio", iso: "PJM", fuel: "solar", capacity_mw: 1500, generation_twh: 3.3, co2_mt: 0 },
            // CAISO - Solar + Battery
            { name: "CAISO Solar Portfolio", iso: "CAISO", fuel: "solar", capacity_mw: 2500, generation_twh: 5.5, co2_mt: 0 },
            { name: "CAISO Battery Storage", iso: "CAISO", fuel: "battery", capacity_mw: 1500, generation_twh: 0, co2_mt: 0 },
            // CAISO - Gas
            { name: "CAISO Gas CCGT Fleet", iso: "CAISO", fuel: "gas_ccgt", capacity_mw: 1200, generation_twh: 6.3, co2_mt: 2.5 },
            // NEISO - Wind
            { name: "NEISO Wind Portfolio", iso: "NEISO", fuel: "wind", capacity_mw: 800, generation_twh: 2.5, co2_mt: 0 },
        ]
    },
    {
        name: "AES Corporation",
        shortName: "AES",
        type: "Hybrid",
        co2_2024_mt: 28,
        intensity_kg_per_mwh: 372,
        total_generation_twh: 75,
        total_capacity_gw: 35,
        target: "Carbon-neutral Scope 1&2 by 2040; SBTi validated 1.5°C",
        pipeline_twh: 25.0, // Large solar+battery pipeline (Fluence partnership) + wind + offshore
        plants: [
            // PJM - Coal
            { name: "Petersburg Coal (IN)", iso: "PJM", fuel: "coal", capacity_mw: 1700, generation_twh: 5.3, co2_mt: 5.3, heat_rate_btu: 10700 },
            // PJM - Gas
            { name: "PJM Gas CCGT Fleet", iso: "PJM", fuel: "gas_ccgt", capacity_mw: 1800, generation_twh: 9.4, co2_mt: 3.8 },
            { name: "PJM Gas Peaker Fleet", iso: "PJM", fuel: "gas_peaker", capacity_mw: 1200, generation_twh: 1.1, co2_mt: 0.7 },
            // CAISO - Solar + Battery
            { name: "CAISO Solar Portfolio", iso: "CAISO", fuel: "solar", capacity_mw: 3000, generation_twh: 6.6, co2_mt: 0 },
            { name: "CAISO Battery Storage", iso: "CAISO", fuel: "battery", capacity_mw: 2000, generation_twh: 0, co2_mt: 0 },
            // CAISO - Gas
            { name: "AES Alamitos CCGT (CA)", iso: "CAISO", fuel: "gas_ccgt", capacity_mw: 1960, generation_twh: 10.3, co2_mt: 4.1 },
            { name: "AES Redondo/Huntington (CA)", iso: "CAISO", fuel: "gas_peaker", capacity_mw: 1500, generation_twh: 1.3, co2_mt: 0.8 },
            // Wind
            { name: "Wind Portfolio (Multi-ISO)", iso: "PJM", fuel: "wind", capacity_mw: 2500, generation_twh: 7.8, co2_mt: 0 },
            // Solar (other)
            { name: "Solar Portfolio (Other)", iso: "PJM", fuel: "solar", capacity_mw: 2000, generation_twh: 4.4, co2_mt: 0 },
        ]
    },
];

// ============================================================================
// FOSSIL DISPATCH DECLINE CURVES - Track 1 Baseline
// ============================================================================
// Computed from the pipeline's merit-order retirement model (dispatch_utils.py):
//   Coal → Oil → Gas displacement as clean energy increases
//   At 70% threshold, all coal and oil fully retired; only gas remains
//
// P10 = low demand growth (0.5%/yr) - more optimistic for decarbonization
// P50 = medium demand growth (1.0%/yr)
// P90 = high demand growth (2.0%/yr) - more fossil remains

const FOSSIL_DISPATCH_DECLINE = {
    CAISO: {
        baseline_clean_pct: 48.5,
        base_demand_twh: 224,
        coal_cap_twh: 0.0,
        oil_cap_twh: 0.6,
        thresholds: {
            "50":   { year: 2030, p10: { coal: 0, oil: 0, gas: 115.5 }, p50: { coal: 0, oil: 0, gas: 117.7 }, p90: { coal: 0, oil: 0, gas: 122.2 } },
            "55":   { year: 2031, p10: { coal: 0, oil: 0, gas: 104.5 }, p50: { coal: 0, oil: 0, gas: 107.0 }, p90: { coal: 0, oil: 0, gas: 112.1 } },
            "60":   { year: 2033, p10: { coal: 0, oil: 0, gas: 93.8 },  p50: { coal: 0, oil: 0, gas: 97.0 },  p90: { coal: 0, oil: 0, gas: 103.8 } },
            "65":   { year: 2034, p10: { coal: 0, oil: 0, gas: 82.4 },  p50: { coal: 0, oil: 0, gas: 85.7 },  p90: { coal: 0, oil: 0, gas: 92.5 } },
            "70":   { year: 2035, p10: { coal: 0, oil: 0, gas: 71.0 },  p50: { coal: 0, oil: 0, gas: 74.2 },  p90: { coal: 0, oil: 0, gas: 81.0 } },
            "75":   { year: 2036, p10: { coal: 0, oil: 0, gas: 59.6 },  p50: { coal: 0, oil: 0, gas: 62.5 },  p90: { coal: 0, oil: 0, gas: 68.8 } },
            "80":   { year: 2037, p10: { coal: 0, oil: 0, gas: 47.9 },  p50: { coal: 0, oil: 0, gas: 50.5 },  p90: { coal: 0, oil: 0, gas: 56.0 } },
            "85":   { year: 2038, p10: { coal: 0, oil: 0, gas: 36.1 },  p50: { coal: 0, oil: 0, gas: 38.2 },  p90: { coal: 0, oil: 0, gas: 42.8 } },
            "87.5": { year: 2039, p10: { coal: 0, oil: 0, gas: 30.2 },  p50: { coal: 0, oil: 0, gas: 32.2 },  p90: { coal: 0, oil: 0, gas: 36.3 } },
            "90":   { year: 2040, p10: { coal: 0, oil: 0, gas: 24.2 },  p50: { coal: 0, oil: 0, gas: 26.0 },  p90: { coal: 0, oil: 0, gas: 29.5 } },
            "92.5": { year: 2043, p10: { coal: 0, oil: 0, gas: 18.1 },  p50: { coal: 0, oil: 0, gas: 20.1 },  p90: { coal: 0, oil: 0, gas: 23.8 } },
            "95":   { year: 2045, p10: { coal: 0, oil: 0, gas: 12.0 },  p50: { coal: 0, oil: 0, gas: 13.7 },  p90: { coal: 0, oil: 0, gas: 17.0 } },
            "97.5": { year: 2048, p10: { coal: 0, oil: 0, gas: 5.9 },   p50: { coal: 0, oil: 0, gas: 7.0 },   p90: { coal: 0, oil: 0, gas: 9.3 } },
            "99":   { year: 2049, p10: { coal: 0, oil: 0, gas: 2.3 },   p50: { coal: 0, oil: 0, gas: 2.8 },   p90: { coal: 0, oil: 0, gas: 3.8 } },
            "99.99":  { year: 2050, p10: { coal: 0, oil: 0, gas: 0 },     p50: { coal: 0, oil: 0, gas: 0 },     p90: { coal: 0, oil: 0, gas: 0 } },
        }
    },
    ERCOT: {
        baseline_clean_pct: 46.1,
        base_demand_twh: 488,
        coal_cap_twh: 67.6,
        oil_cap_twh: 0.0,
        thresholds: {
            "50":   { year: 2030, p10: { coal: 48.9, oil: 0, gas: 203.3 }, p50: { coal: 51.0, oil: 0, gas: 205.4 }, p90: { coal: 55.3, oil: 0, gas: 209.8 } },
            "55":   { year: 2031, p10: { coal: 22.4, oil: 0, gas: 205.3 }, p50: { coal: 25.6, oil: 0, gas: 207.5 }, p90: { coal: 32.2, oil: 0, gas: 212.2 } },
            "60":   { year: 2033, p10: { coal: 0, oil: 0, gas: 203.9 },    p50: { coal: 0, oil: 0, gas: 211.4 },    p90: { coal: 0, oil: 0, gas: 226.8 } },
            "65":   { year: 2034, p10: { coal: 0, oil: 0, gas: 179.5 },    p50: { coal: 0, oil: 0, gas: 186.8 },    p90: { coal: 0, oil: 0, gas: 201.9 } },
            "70":   { year: 2035, p10: { coal: 0, oil: 0, gas: 154.6 },    p50: { coal: 0, oil: 0, gas: 161.7 },    p90: { coal: 0, oil: 0, gas: 176.6 } },
            "75":   { year: 2036, p10: { coal: 0, oil: 0, gas: 129.6 },    p50: { coal: 0, oil: 0, gas: 136.1 },    p90: { coal: 0, oil: 0, gas: 149.8 } },
            "80":   { year: 2037, p10: { coal: 0, oil: 0, gas: 104.4 },    p50: { coal: 0, oil: 0, gas: 110.0 },    p90: { coal: 0, oil: 0, gas: 121.9 } },
            "85":   { year: 2038, p10: { coal: 0, oil: 0, gas: 78.7 },     p50: { coal: 0, oil: 0, gas: 83.3 },     p90: { coal: 0, oil: 0, gas: 93.1 } },
            "87.5": { year: 2039, p10: { coal: 0, oil: 0, gas: 65.9 },     p50: { coal: 0, oil: 0, gas: 70.1 },     p90: { coal: 0, oil: 0, gas: 78.9 } },
            "90":   { year: 2040, p10: { coal: 0, oil: 0, gas: 52.8 },     p50: { coal: 0, oil: 0, gas: 56.7 },     p90: { coal: 0, oil: 0, gas: 64.3 } },
            "92.5": { year: 2043, p10: { coal: 0, oil: 0, gas: 39.4 },     p50: { coal: 0, oil: 0, gas: 43.8 },     p90: { coal: 0, oil: 0, gas: 51.9 } },
            "95":   { year: 2045, p10: { coal: 0, oil: 0, gas: 26.2 },     p50: { coal: 0, oil: 0, gas: 29.8 },     p90: { coal: 0, oil: 0, gas: 37.0 } },
            "97.5": { year: 2048, p10: { coal: 0, oil: 0, gas: 12.8 },     p50: { coal: 0, oil: 0, gas: 15.3 },     p90: { coal: 0, oil: 0, gas: 20.2 } },
            "99":   { year: 2049, p10: { coal: 0, oil: 0, gas: 5.1 },      p50: { coal: 0, oil: 0, gas: 6.2 },      p90: { coal: 0, oil: 0, gas: 8.3 } },
            "99.99":  { year: 2050, p10: { coal: 0, oil: 0, gas: 0 },        p50: { coal: 0, oil: 0, gas: 0 },        p90: { coal: 0, oil: 0, gas: 0 } },
        }
    },
    PJM: {
        baseline_clean_pct: 40.6,
        base_demand_twh: 843.3,
        coal_cap_twh: 139.1,
        oil_cap_twh: 4.6,
        thresholds: {
            "50":   { year: 2030, p10: { coal: 60.3, oil: 4.6, gas: 372.5 }, p50: { coal: 62.9, oil: 4.8, gas: 375.5 }, p90: { coal: 68.2, oil: 5.2, gas: 381.8 } },
            "55":   { year: 2031, p10: { coal: 14.7, oil: 4.7, gas: 376.5 }, p50: { coal: 18.7, oil: 4.9, gas: 379.2 }, p90: { coal: 27.1, oil: 5.3, gas: 384.9 } },
            "60":   { year: 2033, p10: { coal: 0, oil: 0, gas: 351.3 },      p50: { coal: 0, oil: 0, gas: 365.3 },      p90: { coal: 0, oil: 0, gas: 394.3 } },
            "65":   { year: 2034, p10: { coal: 0, oil: 0, gas: 309.8 },      p50: { coal: 0, oil: 0, gas: 322.8 },      p90: { coal: 0, oil: 0, gas: 349.5 } },
            "70":   { year: 2035, p10: { coal: 0, oil: 0, gas: 267.3 },      p50: { coal: 0, oil: 0, gas: 279.5 },      p90: { coal: 0, oil: 0, gas: 305.1 } },
            "75":   { year: 2036, p10: { coal: 0, oil: 0, gas: 224.0 },      p50: { coal: 0, oil: 0, gas: 235.2 },      p90: { coal: 0, oil: 0, gas: 258.7 } },
            "80":   { year: 2037, p10: { coal: 0, oil: 0, gas: 180.4 },      p50: { coal: 0, oil: 0, gas: 190.1 },      p90: { coal: 0, oil: 0, gas: 210.5 } },
            "85":   { year: 2038, p10: { coal: 0, oil: 0, gas: 136.1 },      p50: { coal: 0, oil: 0, gas: 144.0 },      p90: { coal: 0, oil: 0, gas: 160.8 } },
            "87.5": { year: 2039, p10: { coal: 0, oil: 0, gas: 114.1 },      p50: { coal: 0, oil: 0, gas: 121.2 },      p90: { coal: 0, oil: 0, gas: 136.3 } },
            "90":   { year: 2040, p10: { coal: 0, oil: 0, gas: 91.6 },       p50: { coal: 0, oil: 0, gas: 97.9 },       p90: { coal: 0, oil: 0, gas: 111.1 } },
            "92.5": { year: 2043, p10: { coal: 0, oil: 0, gas: 68.1 },       p50: { coal: 0, oil: 0, gas: 75.7 },       p90: { coal: 0, oil: 0, gas: 89.7 } },
            "95":   { year: 2045, p10: { coal: 0, oil: 0, gas: 45.2 },       p50: { coal: 0, oil: 0, gas: 51.5 },       p90: { coal: 0, oil: 0, gas: 64.0 } },
            "97.5": { year: 2048, p10: { coal: 0, oil: 0, gas: 22.2 },       p50: { coal: 0, oil: 0, gas: 26.5 },       p90: { coal: 0, oil: 0, gas: 35.0 } },
            "99":   { year: 2049, p10: { coal: 0, oil: 0, gas: 8.9 },        p50: { coal: 0, oil: 0, gas: 10.7 },       p90: { coal: 0, oil: 0, gas: 14.3 } },
            "99.99":  { year: 2050, p10: { coal: 0, oil: 0, gas: 0 },          p50: { coal: 0, oil: 0, gas: 0 },          p90: { coal: 0, oil: 0, gas: 0 } },
        }
    },
    NYISO: {
        baseline_clean_pct: 39.0,
        base_demand_twh: 151.6,
        coal_cap_twh: 0.0,
        oil_cap_twh: 0.15,
        thresholds: {
            "50":   { year: 2030, p10: { coal: 0, oil: 0, gas: 78.3 },  p50: { coal: 0, oil: 0, gas: 79.7 },  p90: { coal: 0, oil: 0, gas: 82.8 } },
            "55":   { year: 2031, p10: { coal: 0, oil: 0, gas: 70.7 },  p50: { coal: 0, oil: 0, gas: 72.4 },  p90: { coal: 0, oil: 0, gas: 75.9 } },
            "60":   { year: 2033, p10: { coal: 0, oil: 0, gas: 63.5 },  p50: { coal: 0, oil: 0, gas: 65.7 },  p90: { coal: 0, oil: 0, gas: 70.3 } },
            "65":   { year: 2034, p10: { coal: 0, oil: 0, gas: 55.8 },  p50: { coal: 0, oil: 0, gas: 58.0 },  p90: { coal: 0, oil: 0, gas: 62.7 } },
            "70":   { year: 2035, p10: { coal: 0, oil: 0, gas: 48.0 },  p50: { coal: 0, oil: 0, gas: 50.2 },  p90: { coal: 0, oil: 0, gas: 54.8 } },
            "75":   { year: 2036, p10: { coal: 0, oil: 0, gas: 40.2 },  p50: { coal: 0, oil: 0, gas: 42.3 },  p90: { coal: 0, oil: 0, gas: 46.5 } },
            "80":   { year: 2037, p10: { coal: 0, oil: 0, gas: 32.3 },  p50: { coal: 0, oil: 0, gas: 34.2 },  p90: { coal: 0, oil: 0, gas: 37.9 } },
            "85":   { year: 2038, p10: { coal: 0, oil: 0, gas: 24.3 },  p50: { coal: 0, oil: 0, gas: 25.9 },  p90: { coal: 0, oil: 0, gas: 28.9 } },
            "87.5": { year: 2039, p10: { coal: 0, oil: 0, gas: 20.4 },  p50: { coal: 0, oil: 0, gas: 21.8 },  p90: { coal: 0, oil: 0, gas: 24.5 } },
            "90":   { year: 2040, p10: { coal: 0, oil: 0, gas: 16.3 },  p50: { coal: 0, oil: 0, gas: 17.6 },  p90: { coal: 0, oil: 0, gas: 19.9 } },
            "92.5": { year: 2043, p10: { coal: 0, oil: 0, gas: 12.2 },  p50: { coal: 0, oil: 0, gas: 13.6 },  p90: { coal: 0, oil: 0, gas: 16.1 } },
            "95":   { year: 2045, p10: { coal: 0, oil: 0, gas: 8.1 },   p50: { coal: 0, oil: 0, gas: 9.2 },   p90: { coal: 0, oil: 0, gas: 11.5 } },
            "97.5": { year: 2048, p10: { coal: 0, oil: 0, gas: 4.0 },   p50: { coal: 0, oil: 0, gas: 4.8 },   p90: { coal: 0, oil: 0, gas: 6.3 } },
            "99":   { year: 2049, p10: { coal: 0, oil: 0, gas: 1.6 },   p50: { coal: 0, oil: 0, gas: 1.9 },   p90: { coal: 0, oil: 0, gas: 2.6 } },
            "99.99":  { year: 2050, p10: { coal: 0, oil: 0, gas: 0 },     p50: { coal: 0, oil: 0, gas: 0 },     p90: { coal: 0, oil: 0, gas: 0 } },
        }
    },
    NEISO: {
        baseline_clean_pct: 33.5,
        base_demand_twh: 115.3,
        coal_cap_twh: 0.31,
        oil_cap_twh: 1.29,
        thresholds: {
            "50":   { year: 2030, p10: { coal: 0, oil: 0, gas: 59.4 },  p50: { coal: 0, oil: 0, gas: 60.6 },  p90: { coal: 0, oil: 0, gas: 62.9 } },
            "55":   { year: 2031, p10: { coal: 0, oil: 0, gas: 53.7 },  p50: { coal: 0, oil: 0, gas: 55.1 },  p90: { coal: 0, oil: 0, gas: 57.8 } },
            "60":   { year: 2033, p10: { coal: 0, oil: 0, gas: 48.2 },  p50: { coal: 0, oil: 0, gas: 49.9 },  p90: { coal: 0, oil: 0, gas: 53.4 } },
            "65":   { year: 2034, p10: { coal: 0, oil: 0, gas: 42.3 },  p50: { coal: 0, oil: 0, gas: 44.1 },  p90: { coal: 0, oil: 0, gas: 47.7 } },
            "70":   { year: 2035, p10: { coal: 0, oil: 0, gas: 36.5 },  p50: { coal: 0, oil: 0, gas: 38.2 },  p90: { coal: 0, oil: 0, gas: 41.7 } },
            "75":   { year: 2036, p10: { coal: 0, oil: 0, gas: 30.6 },  p50: { coal: 0, oil: 0, gas: 32.2 },  p90: { coal: 0, oil: 0, gas: 35.4 } },
            "80":   { year: 2037, p10: { coal: 0, oil: 0, gas: 24.6 },  p50: { coal: 0, oil: 0, gas: 26.0 },  p90: { coal: 0, oil: 0, gas: 28.8 } },
            "85":   { year: 2038, p10: { coal: 0, oil: 0, gas: 18.5 },  p50: { coal: 0, oil: 0, gas: 19.7 },  p90: { coal: 0, oil: 0, gas: 22.0 } },
            "87.5": { year: 2039, p10: { coal: 0, oil: 0, gas: 15.5 },  p50: { coal: 0, oil: 0, gas: 16.6 },  p90: { coal: 0, oil: 0, gas: 18.7 } },
            "90":   { year: 2040, p10: { coal: 0, oil: 0, gas: 12.5 },  p50: { coal: 0, oil: 0, gas: 13.4 },  p90: { coal: 0, oil: 0, gas: 15.2 } },
            "92.5": { year: 2043, p10: { coal: 0, oil: 0, gas: 9.3 },   p50: { coal: 0, oil: 0, gas: 10.3 },  p90: { coal: 0, oil: 0, gas: 12.3 } },
            "95":   { year: 2045, p10: { coal: 0, oil: 0, gas: 6.2 },   p50: { coal: 0, oil: 0, gas: 7.0 },   p90: { coal: 0, oil: 0, gas: 8.7 } },
            "97.5": { year: 2048, p10: { coal: 0, oil: 0, gas: 3.0 },   p50: { coal: 0, oil: 0, gas: 3.6 },   p90: { coal: 0, oil: 0, gas: 4.8 } },
            "99":   { year: 2049, p10: { coal: 0, oil: 0, gas: 1.2 },   p50: { coal: 0, oil: 0, gas: 1.5 },   p90: { coal: 0, oil: 0, gas: 2.0 } },
            "99.99":  { year: 2050, p10: { coal: 0, oil: 0, gas: 0 },     p50: { coal: 0, oil: 0, gas: 0 },     p90: { coal: 0, oil: 0, gas: 0 } },
        }
    }
};

// ============================================================================
// FUEL COLORS AND LABELS
// ============================================================================
const FUEL_COLORS = {
    coal: '#374151',
    gas_ccgt: '#EF4444',
    gas_peaker: '#FB923C',
    oil: '#9CA3AF',
    nuclear: '#7C3AED',
    solar: '#F59E0B',
    wind: '#22C55E',
    battery: '#06B6D4',
    hydro: '#0EA5E9',
    geothermal: '#10B981',
};

const FUEL_LABELS = {
    coal: 'Coal',
    gas_ccgt: 'Gas CCGT',
    gas_peaker: 'Gas Peaker',
    oil: 'Oil',
    nuclear: 'Nuclear',
    solar: 'Solar',
    wind: 'Wind',
    battery: 'Battery',
    hydro: 'Hydro',
    geothermal: 'Geothermal',
};

const FOSSIL_FUELS = ['coal', 'oil', 'gas_ccgt', 'gas_peaker'];
const CLEAN_FUELS = ['nuclear', 'solar', 'wind', 'battery', 'hydro', 'geothermal'];
const ISO_LIST = ['PJM', 'ERCOT', 'CAISO', 'NYISO', 'NEISO'];

// ============================================================================
// EMISSION RATES (tCO2/MWh by fuel type, from eGRID)
// ============================================================================
const EMISSION_RATES_TCO2_MWH = {
    CAISO: { coal: 0.514, gas: 0.391, oil: 0.818 },
    ERCOT: { coal: 1.055, gas: 0.393, oil: 1.313 },
    PJM:   { coal: 1.005, gas: 0.393, oil: 0.871 },
    NYISO: { coal: 0,     gas: 0.415, oil: 0.433 },
    NEISO: { coal: 1.043, gas: 0.383, oil: 0.999 },
};

// ============================================================================
// HELPER: Compute company survival at each threshold
// ============================================================================
/**
 * For a given company and set of fuel filters, compute the "survival rate"
 * at each threshold - the percentage of their fossil capacity that still
 * has dispatch room in the grid.
 *
 * Logic:
 *   - At each threshold, we know how much coal/oil/gas TWh remain in each ISO
 *   - For gas, CCGTs dispatch before peakers (lower heat rate)
 *   - If total remaining gas TWh < all CCGT generation in that ISO, peakers
 *     are fully curtailed and CCGTs partially curtailed
 *   - A plant's survival = min(1, iso_fuel_remaining_twh / iso_fuel_total_twh)
 *
 * Returns: { thresholds: [...], years: [...],
 *            survival_pct: { p10: [...], p50: [...], p90: [...] },
 *            co2_remaining_mt: { p10: [...], p50: [...], p90: [...] },
 *            co2_reduction_pct: { p10: [...], p50: [...], p90: [...] } }
 */
function computeFleetSurvival(company, fuelFilter) {
    fuelFilter = fuelFilter || 'all_fossil'; // 'all_fossil', 'coal', 'oil', 'gas', 'gas_ccgt', 'gas_peaker', 'all'

    const thresholds = [];
    const years = [];
    const survival_p10 = [], survival_p50 = [], survival_p90 = [];
    const co2_remaining_p10 = [], co2_remaining_p50 = [], co2_remaining_p90 = [];
    const co2_reduction_p10 = [], co2_reduction_p50 = [], co2_reduction_p90 = [];

    // Compute baseline totals
    let baseline_fossil_capacity = 0;
    let baseline_fossil_gen = 0;
    let baseline_co2 = 0;
    let baseline_total_co2 = company.co2_2024_mt;

    // Categorize plants
    const plants = company.plants.filter(p => {
        if (fuelFilter === 'all') return true;
        if (fuelFilter === 'all_fossil') return FOSSIL_FUELS.includes(p.fuel);
        if (fuelFilter === 'coal') return p.fuel === 'coal';
        if (fuelFilter === 'oil') return p.fuel === 'oil';
        if (fuelFilter === 'gas') return p.fuel === 'gas_ccgt' || p.fuel === 'gas_peaker';
        if (fuelFilter === 'gas_ccgt') return p.fuel === 'gas_ccgt';
        if (fuelFilter === 'gas_peaker') return p.fuel === 'gas_peaker';
        return false;
    });

    plants.forEach(p => {
        if (FOSSIL_FUELS.includes(p.fuel)) {
            baseline_fossil_capacity += p.capacity_mw;
            baseline_fossil_gen += p.generation_twh;
            baseline_co2 += p.co2_mt;
        }
    });

    if (baseline_fossil_capacity === 0 && fuelFilter !== 'all') {
        // No fossil plants matching filter - survival is N/A
        FLEET_THRESHOLDS.forEach(t => {
            thresholds.push(t);
            years.push(FLEET_THRESHOLD_YEARS[t]);
            survival_p10.push(fuelFilter === 'all' ? 100 : null);
            survival_p50.push(fuelFilter === 'all' ? 100 : null);
            survival_p90.push(fuelFilter === 'all' ? 100 : null);
            co2_remaining_p10.push(0);
            co2_remaining_p50.push(0);
            co2_remaining_p90.push(0);
            co2_reduction_p10.push(100);
            co2_reduction_p50.push(100);
            co2_reduction_p90.push(100);
        });
        return { thresholds, years, survival_pct: { p10: survival_p10, p50: survival_p50, p90: survival_p90 },
                 co2_remaining_mt: { p10: co2_remaining_p10, p50: co2_remaining_p50, p90: co2_remaining_p90 },
                 co2_reduction_pct: { p10: co2_reduction_p10, p50: co2_reduction_p50, p90: co2_reduction_p90 },
                 baseline_fossil_capacity, baseline_fossil_gen, baseline_co2 };
    }

    FLEET_THRESHOLDS.forEach(t => {
        thresholds.push(t);
        years.push(FLEET_THRESHOLD_YEARS[t]);

        ['p10', 'p50', 'p90'].forEach(pct => {
            let surviving_capacity = 0;
            let surviving_co2 = 0;

            plants.forEach(plant => {
                const iso = plant.iso;
                const decline = FOSSIL_DISPATCH_DECLINE[iso];
                if (!decline) {
                    // ISO not in our model - assume survival
                    surviving_capacity += plant.capacity_mw;
                    surviving_co2 += plant.co2_mt;
                    return;
                }

                const tData = decline.thresholds[String(t)];
                if (!tData) return;

                const remaining = tData[pct];
                const base = decline.base_demand_twh;

                if (CLEAN_FUELS.includes(plant.fuel)) {
                    // Clean plants always survive
                    surviving_capacity += plant.capacity_mw;
                    return;
                }

                if (plant.fuel === 'coal') {
                    const coal_remaining = remaining.coal;
                    const coal_cap = decline.coal_cap_twh;
                    const survival_frac = coal_cap > 0 ? Math.min(1, coal_remaining / coal_cap) : 0;
                    surviving_capacity += plant.capacity_mw * survival_frac;
                    surviving_co2 += plant.co2_mt * survival_frac;
                } else if (plant.fuel === 'oil') {
                    const oil_remaining = remaining.oil || 0;
                    const oil_cap = decline.oil_cap_twh;
                    const survival_frac = oil_cap > 0 ? Math.min(1, oil_remaining / oil_cap) : 0;
                    surviving_capacity += plant.capacity_mw * survival_frac;
                    surviving_co2 += plant.co2_mt * survival_frac;
                } else if (plant.fuel === 'gas_ccgt' || plant.fuel === 'gas_peaker') {
                    // Gas dispatch priority: CCGTs first, then peakers
                    const gas_remaining_twh = remaining.gas;
                    // Estimate total CCGT and peaker gen in this ISO for this company
                    const iso_ccgt_gen = company.plants
                        .filter(p => p.iso === iso && p.fuel === 'gas_ccgt')
                        .reduce((s, p) => s + p.generation_twh, 0);
                    const iso_peaker_gen = company.plants
                        .filter(p => p.iso === iso && p.fuel === 'gas_peaker')
                        .reduce((s, p) => s + p.generation_twh, 0);
                    const iso_total_gas_gen = iso_ccgt_gen + iso_peaker_gen;

                    // Get total regional gas generation at baseline
                    const baseline_fossil_pct = (100 - decline.baseline_clean_pct) / 100;
                    const total_regional_gas = base * baseline_fossil_pct - decline.coal_cap_twh - decline.oil_cap_twh;

                    // What fraction of regional gas remains?
                    const regional_gas_frac = total_regional_gas > 0 ?
                        Math.min(1, gas_remaining_twh / total_regional_gas) : 0;

                    if (plant.fuel === 'gas_ccgt') {
                        // CCGTs survive longer - they're dispatched first
                        // If gas fraction > (peaker share of gas), CCGTs fully survive
                        const peaker_share = iso_total_gas_gen > 0 ? iso_peaker_gen / iso_total_gas_gen : 0;
                        const ccgt_survival = peaker_share < 1 ?
                            Math.min(1, regional_gas_frac / (1 - peaker_share)) : regional_gas_frac;
                        surviving_capacity += plant.capacity_mw * Math.min(1, ccgt_survival);
                        surviving_co2 += plant.co2_mt * Math.min(1, ccgt_survival);
                    } else {
                        // Peakers curtailed first
                        const ccgt_share = iso_total_gas_gen > 0 ? iso_ccgt_gen / iso_total_gas_gen : 0;
                        const after_ccgt = Math.max(0, regional_gas_frac - ccgt_share);
                        const peaker_survival = ccgt_share < 1 ?
                            Math.min(1, after_ccgt / (1 - ccgt_share)) : 0;
                        surviving_capacity += plant.capacity_mw * Math.min(1, peaker_survival);
                        surviving_co2 += plant.co2_mt * Math.min(1, peaker_survival);
                    }
                }
            });

            const survival_rate = baseline_fossil_capacity > 0 ?
                (surviving_capacity / baseline_fossil_capacity) * 100 : 100;
            const co2_reduction = baseline_co2 > 0 ?
                ((baseline_co2 - surviving_co2) / baseline_co2) * 100 : 100;

            if (pct === 'p10') {
                survival_p10.push(Math.round(survival_rate * 10) / 10);
                co2_remaining_p10.push(Math.round(surviving_co2 * 100) / 100);
                co2_reduction_p10.push(Math.round(co2_reduction * 10) / 10);
            } else if (pct === 'p50') {
                survival_p50.push(Math.round(survival_rate * 10) / 10);
                co2_remaining_p50.push(Math.round(surviving_co2 * 100) / 100);
                co2_reduction_p50.push(Math.round(co2_reduction * 10) / 10);
            } else {
                survival_p90.push(Math.round(survival_rate * 10) / 10);
                co2_remaining_p90.push(Math.round(surviving_co2 * 100) / 100);
                co2_reduction_p90.push(Math.round(co2_reduction * 10) / 10);
            }
        });
    });

    return {
        thresholds, years,
        survival_pct: { p10: survival_p10, p50: survival_p50, p90: survival_p90 },
        co2_remaining_mt: { p10: co2_remaining_p10, p50: co2_remaining_p50, p90: co2_remaining_p90 },
        co2_reduction_pct: { p10: co2_reduction_p10, p50: co2_reduction_p50, p90: co2_reduction_p90 },
        baseline_fossil_capacity, baseline_fossil_gen, baseline_co2,
    };
}

/**
 * Compute the full fleet mix (all fuel types) at each threshold with P10/P50/P90 for fossil.
 * Clean assets stay constant; fossil assets decline per the dispatch model.
 *
 * Returns: {
 *   thresholds, years,
 *   capacity_gw: { <fuel>: { p10: [...], p50: [...], p90: [...] } },
 *   generation_twh: { <fuel>: { p10: [...], p50: [...], p90: [...] } },
 *   total_capacity_gw: { p10: [...], p50: [...], p90: [...] },
 *   total_generation_twh: { p10: [...], p50: [...], p90: [...] },
 *   baseline_total_capacity_gw, baseline_total_generation_twh,
 * }
 */
function computeFleetMixOverTime(company) {
    const thresholds = [];
    const years = [];
    const allFuels = [...new Set(company.plants.map(p => p.fuel))];

    // Initialize per-fuel arrays
    const capacity_gw = {};
    const generation_twh = {};
    allFuels.forEach(f => {
        capacity_gw[f] = { p10: [], p50: [], p90: [] };
        generation_twh[f] = { p10: [], p50: [], p90: [] };
    });
    const total_capacity_gw = { p10: [], p50: [], p90: [] };
    const total_generation_twh = { p10: [], p50: [], p90: [] };

    // Baseline totals
    let baseline_total_cap = 0, baseline_total_gen = 0;
    company.plants.forEach(p => {
        baseline_total_cap += p.capacity_mw;
        baseline_total_gen += p.generation_twh;
    });

    FLEET_THRESHOLDS.forEach(t => {
        thresholds.push(t);
        years.push(FLEET_THRESHOLD_YEARS[t]);

        ['p10', 'p50', 'p90'].forEach(pct => {
            const fuel_cap = {};
            const fuel_gen = {};
            allFuels.forEach(f => { fuel_cap[f] = 0; fuel_gen[f] = 0; });

            company.plants.forEach(plant => {
                const iso = plant.iso;
                const decline = FOSSIL_DISPATCH_DECLINE[iso];

                if (CLEAN_FUELS.includes(plant.fuel)) {
                    // Clean plants always survive at 100%
                    fuel_cap[plant.fuel] += plant.capacity_mw;
                    fuel_gen[plant.fuel] += plant.generation_twh;
                    return;
                }

                if (!decline) {
                    fuel_cap[plant.fuel] += plant.capacity_mw;
                    fuel_gen[plant.fuel] += plant.generation_twh;
                    return;
                }

                const tData = decline.thresholds[String(t)];
                if (!tData) return;
                const remaining = tData[pct];
                const base = decline.base_demand_twh;

                if (plant.fuel === 'coal') {
                    const coal_cap = decline.coal_cap_twh;
                    const frac = coal_cap > 0 ? Math.min(1, remaining.coal / coal_cap) : 0;
                    fuel_cap[plant.fuel] += plant.capacity_mw * frac;
                    fuel_gen[plant.fuel] += plant.generation_twh * frac;
                } else if (plant.fuel === 'oil') {
                    const oil_cap = decline.oil_cap_twh;
                    const frac = oil_cap > 0 ? Math.min(1, (remaining.oil || 0) / oil_cap) : 0;
                    fuel_cap[plant.fuel] += plant.capacity_mw * frac;
                    fuel_gen[plant.fuel] += plant.generation_twh * frac;
                } else if (plant.fuel === 'gas_ccgt' || plant.fuel === 'gas_peaker') {
                    const gas_remaining_twh = remaining.gas;
                    const iso_ccgt_gen = company.plants
                        .filter(p => p.iso === iso && p.fuel === 'gas_ccgt')
                        .reduce((s, p) => s + p.generation_twh, 0);
                    const iso_peaker_gen = company.plants
                        .filter(p => p.iso === iso && p.fuel === 'gas_peaker')
                        .reduce((s, p) => s + p.generation_twh, 0);
                    const iso_total_gas_gen = iso_ccgt_gen + iso_peaker_gen;
                    const baseline_fossil_pct = (100 - decline.baseline_clean_pct) / 100;
                    const total_regional_gas = base * baseline_fossil_pct - decline.coal_cap_twh - decline.oil_cap_twh;
                    const regional_gas_frac = total_regional_gas > 0 ?
                        Math.min(1, gas_remaining_twh / total_regional_gas) : 0;

                    if (plant.fuel === 'gas_ccgt') {
                        const peaker_share = iso_total_gas_gen > 0 ? iso_peaker_gen / iso_total_gas_gen : 0;
                        const ccgt_survival = peaker_share < 1 ?
                            Math.min(1, regional_gas_frac / (1 - peaker_share)) : regional_gas_frac;
                        const frac = Math.min(1, ccgt_survival);
                        fuel_cap[plant.fuel] += plant.capacity_mw * frac;
                        fuel_gen[plant.fuel] += plant.generation_twh * frac;
                    } else {
                        const ccgt_share = iso_total_gas_gen > 0 ? iso_ccgt_gen / iso_total_gas_gen : 0;
                        const after_ccgt = Math.max(0, regional_gas_frac - ccgt_share);
                        const peaker_survival = ccgt_share < 1 ?
                            Math.min(1, after_ccgt / (1 - ccgt_share)) : 0;
                        const frac = Math.min(1, peaker_survival);
                        fuel_cap[plant.fuel] += plant.capacity_mw * frac;
                        fuel_gen[plant.fuel] += plant.generation_twh * frac;
                    }
                }
            });

            // Store per-fuel results
            allFuels.forEach(f => {
                capacity_gw[f][pct].push(Math.round(fuel_cap[f] / 10) / 100); // MW → GW, 2 decimal
                generation_twh[f][pct].push(Math.round(fuel_gen[f] * 100) / 100);
            });

            // Store totals
            let tot_cap = 0, tot_gen = 0;
            allFuels.forEach(f => { tot_cap += fuel_cap[f]; tot_gen += fuel_gen[f]; });
            total_capacity_gw[pct].push(Math.round(tot_cap / 10) / 100);
            total_generation_twh[pct].push(Math.round(tot_gen * 100) / 100);
        });
    });

    return {
        thresholds, years, fuels: allFuels,
        capacity_gw, generation_twh,
        total_capacity_gw, total_generation_twh,
        baseline_total_capacity_gw: Math.round(baseline_total_cap / 10) / 100,
        baseline_total_generation_twh: Math.round(baseline_total_gen * 100) / 100,
    };
}

/**
 * Compute milestone years for a company's fleet: when each fossil fuel type's
 * dispatch drops below meaningful thresholds.
 *
 * Returns: {
 *   coal_exit_year: year when company coal dispatch → 0 (or null if no coal),
 *   oil_exit_year: year when company oil dispatch → 0 (or null),
 *   peaker_squeeze_year: year when peaker dispatch drops below 50% of baseline,
 *   ccgt_erosion_year: year when CCGT dispatch drops below 90% of baseline
 *                       (the start of real CCGT pain - lower CFs, not retirements),
 * }
 */
function computeFleetMilestones(company) {
    const mix = computeFleetMixOverTime(company);
    const result = { coal_exit_year: null, oil_exit_year: null, peaker_squeeze_year: null, ccgt_erosion_year: null };

    // Check if company has each fuel type
    const hasCoal = mix.fuels.includes('coal') && mix.generation_twh.coal && mix.generation_twh.coal.p50[0] > 0.01;
    const hasOil = mix.fuels.includes('oil') && mix.generation_twh.oil && mix.generation_twh.oil.p50[0] > 0.01;
    const hasPeaker = mix.fuels.includes('gas_peaker') && mix.generation_twh.gas_peaker;
    const hasCCGT = mix.fuels.includes('gas_ccgt') && mix.generation_twh.gas_ccgt;

    for (let i = 0; i < mix.thresholds.length; i++) {
        const yr = mix.years[i];

        // Coal exit: first year coal TWh → 0
        if (hasCoal && !result.coal_exit_year && mix.generation_twh.coal.p50[i] < 0.01) {
            result.coal_exit_year = yr;
        }
        // Oil exit: first year oil TWh → 0
        if (hasOil && !result.oil_exit_year && mix.generation_twh.oil.p50[i] < 0.01) {
            result.oil_exit_year = yr;
        }
        // Peaker squeeze: first year peaker gen < 50% of baseline
        if (hasPeaker && !result.peaker_squeeze_year) {
            const baseline = mix.generation_twh.gas_peaker.p50[0];
            if (baseline > 0 && mix.generation_twh.gas_peaker.p50[i] < baseline * 0.5) {
                result.peaker_squeeze_year = yr;
            }
        }
        // CCGT erosion: first year CCGT gen < 90% of baseline (CFs start dropping meaningfully)
        if (hasCCGT && !result.ccgt_erosion_year) {
            const baseline = mix.generation_twh.gas_ccgt.p50[0];
            if (baseline > 0 && mix.generation_twh.gas_ccgt.p50[i] < baseline * 0.9) {
                result.ccgt_erosion_year = yr;
            }
        }
    }

    return result;
}

/**
 * Get aggregated fleet stats for a company by ISO and fuel type.
 */
function getFleetBreakdown(company) {
    const byISO = {};
    const byFuel = {};
    let total_capacity = 0, total_gen = 0, total_co2 = 0;
    let fossil_capacity = 0, fossil_gen = 0, fossil_co2 = 0;
    let clean_capacity = 0, clean_gen = 0;

    company.plants.forEach(p => {
        // By ISO
        if (!byISO[p.iso]) byISO[p.iso] = { capacity_mw: 0, generation_twh: 0, co2_mt: 0, plants: [] };
        byISO[p.iso].capacity_mw += p.capacity_mw;
        byISO[p.iso].generation_twh += p.generation_twh;
        byISO[p.iso].co2_mt += p.co2_mt;
        byISO[p.iso].plants.push(p);

        // By fuel
        if (!byFuel[p.fuel]) byFuel[p.fuel] = { capacity_mw: 0, generation_twh: 0, co2_mt: 0 };
        byFuel[p.fuel].capacity_mw += p.capacity_mw;
        byFuel[p.fuel].generation_twh += p.generation_twh;
        byFuel[p.fuel].co2_mt += p.co2_mt;

        total_capacity += p.capacity_mw;
        total_gen += p.generation_twh;
        total_co2 += p.co2_mt;

        if (FOSSIL_FUELS.includes(p.fuel)) {
            fossil_capacity += p.capacity_mw;
            fossil_gen += p.generation_twh;
            fossil_co2 += p.co2_mt;
        } else {
            clean_capacity += p.capacity_mw;
            clean_gen += p.generation_twh;
        }
    });

    return {
        byISO, byFuel,
        total_capacity, total_gen, total_co2,
        fossil_capacity, fossil_gen, fossil_co2,
        clean_capacity, clean_gen,
        fossil_pct: total_capacity > 0 ? (fossil_capacity / total_capacity * 100) : 0,
        clean_pct: total_capacity > 0 ? (clean_capacity / total_capacity * 100) : 0,
    };
}

// ============================================================================
// REGIONAL FOSSIL FLEET - Nameplate Capacity & Dispatch Model
// ============================================================================
// Sources:
//   - Installed capacity: EIA Form 860 (2024), PP6 LMP module estimates
//   - Capacity shares: PP6 FOSSIL_CAPACITY_SHARES
//   - Coal retirements: EIA planned retirements + company announcements (Vistra, NRG, Talen, etc.)
//   - Dispatch decline: FOSSIL_DISPATCH_DECLINE above (Step 4/PP4 merit-order)

const REGIONAL_FOSSIL_MW = {
    PJM:   { total: 127800, coal: 37062, gas_ccgt: 47286, gas_ct: 39618, oil: 3834 },
    ERCOT: { total: 80000,  coal: 17600, gas_ccgt: 40000, gas_ct: 22400, oil: 0 },
    CAISO: { total: 47000,  coal: 0,     gas_ccgt: 25850, gas_ct: 18800, oil: 2350 },
    NYISO: { total: 28000,  coal: 0,     gas_ccgt: 13440, gas_ct: 11200, oil: 3360 },
    NEISO: { total: 16000,  coal: 640,   gas_ccgt: 8000,  gas_ct: 6400,  oil: 960 },
};

// Baseline CCGT share of total gas dispatch (calibrated from EIA capacity factors)
const CCGT_GAS_DISPATCH_SHARE = {
    PJM: 0.79, ERCOT: 0.80, CAISO: 0.78, NYISO: 0.75, NEISO: 0.80
};

// Cumulative coal MW retired by year [year, cumulative_mw]
// Sources: Vistra (Martin Lake, Coleto Creek, IL fleet), NRG (Parish, Limestone),
//          Talen (Brandon Shores RMR→2029, Brunner Island), EPA GHG rule compliance
const COAL_RETIREMENT_SCHEDULE = {
    PJM:   [[2026, 3000], [2027, 8000], [2028, 12000], [2030, 20000], [2033, 37062]],
    ERCOT: [[2027, 4500], [2028, 7500], [2030, 12500], [2033, 17600]],
    CAISO: [],
    NYISO: [],
    NEISO: [[2026, 640]],
};

/**
 * Compute regional fossil fleet metrics at each SBTi threshold (P50 only).
 * Returns { thresholds, years, nameplate_gw, dispatch_twh, capacity_factor_pct, ... }
 * Each metric is keyed by fuel type: coal, gas_ccgt, gas_ct, oil, total.
 */
function computeRegionalFossilMetrics(iso) {
    const decline = FOSSIL_DISPATCH_DECLINE[iso];
    const cap = REGIONAL_FOSSIL_MW[iso];
    if (!decline || !cap) return null;

    const retirements = COAL_RETIREMENT_SCHEDULE[iso] || [];
    const ccgtShare = CCGT_GAS_DISPATCH_SHARE[iso] || 0.78;

    // Baseline gas dispatch from regional data
    const fossilFrac = (100 - decline.baseline_clean_pct) / 100;
    const baselineFossilTwh = decline.base_demand_twh * fossilFrac;
    const baselineGasTwh = baselineFossilTwh - decline.coal_cap_twh - (decline.oil_cap_twh || 0);
    const baselineCcgtTwh = baselineGasTwh * ccgtShare;
    const baselineCtTwh = baselineGasTwh * (1 - ccgtShare);

    const out = {
        iso,
        thresholds: [], years: [],
        nameplate_gw: { coal: [], gas_ccgt: [], gas_ct: [], oil: [], total: [] },
        dispatch_twh: { coal: [], gas_ccgt: [], gas_ct: [], oil: [], total: [] },
        capacity_factor_pct: { coal: [], gas_ccgt: [], gas_ct: [], oil: [] },
        baseline_gas_twh: baselineGasTwh,
        baseline_ccgt_twh: baselineCcgtTwh,
        baseline_ct_twh: baselineCtTwh,
    };

    FLEET_THRESHOLDS.forEach(t => {
        const year = FLEET_THRESHOLD_YEARS[t];
        const d = decline.thresholds[String(t)];
        if (!d) return;

        out.thresholds.push(t);
        out.years.push(year);

        // --- Nameplate capacity (coal declines with retirements, gas stays flat) ---
        let coalRetired = 0;
        retirements.forEach(([yr, cum]) => { if (yr <= year) coalRetired = Math.max(coalRetired, cum); });
        const npCoal = Math.max(0, cap.coal - coalRetired);
        const npCcgt = cap.gas_ccgt;
        const npCt = cap.gas_ct;
        const npOil = cap.oil;

        out.nameplate_gw.coal.push(+(npCoal / 1000).toFixed(2));
        out.nameplate_gw.gas_ccgt.push(+(npCcgt / 1000).toFixed(2));
        out.nameplate_gw.gas_ct.push(+(npCt / 1000).toFixed(2));
        out.nameplate_gw.oil.push(+(npOil / 1000).toFixed(2));
        out.nameplate_gw.total.push(+((npCoal + npCcgt + npCt + npOil) / 1000).toFixed(2));

        // --- Dispatch (merit-order split: CCGT first, then CT) ---
        const coalDisp = d.p50.coal;
        const oilDisp = d.p50.oil || 0;
        const totalGas = d.p50.gas;

        let ccgtDisp, ctDisp;
        if (totalGas >= baselineGasTwh) {
            const excess = totalGas - baselineGasTwh;
            ccgtDisp = baselineCcgtTwh + excess * 0.7;
            ctDisp = baselineCtTwh + excess * 0.3;
        } else if (totalGas >= baselineCcgtTwh) {
            ccgtDisp = baselineCcgtTwh;
            ctDisp = totalGas - baselineCcgtTwh;
        } else {
            ccgtDisp = totalGas;
            ctDisp = 0;
        }

        out.dispatch_twh.coal.push(+coalDisp.toFixed(1));
        out.dispatch_twh.gas_ccgt.push(+ccgtDisp.toFixed(1));
        out.dispatch_twh.gas_ct.push(+ctDisp.toFixed(1));
        out.dispatch_twh.oil.push(+oilDisp.toFixed(1));
        out.dispatch_twh.total.push(+(coalDisp + ccgtDisp + ctDisp + oilDisp).toFixed(1));

        // --- Capacity Factors (%) ---
        const cfPct = (twh, mw) => mw > 0 ? +((twh * 1e6) / (mw * 8760) * 100).toFixed(1) : 0;
        out.capacity_factor_pct.coal.push(cfPct(coalDisp, npCoal));
        out.capacity_factor_pct.gas_ccgt.push(cfPct(ccgtDisp, npCcgt));
        out.capacity_factor_pct.gas_ct.push(cfPct(ctDisp, npCt));
        out.capacity_factor_pct.oil.push(cfPct(oilDisp, npOil));
    });

    return out;
}

/**
 * Compute aggregate (all ISOs) fossil fleet metrics.
 */
function computeAggregateFossilMetrics() {
    const isoData = {};
    ISO_LIST.forEach(iso => { isoData[iso] = computeRegionalFossilMetrics(iso); });

    const agg = {
        iso: 'ALL',
        thresholds: FLEET_THRESHOLDS.slice(),
        years: FLEET_THRESHOLDS.map(t => FLEET_THRESHOLD_YEARS[t]),
        nameplate_gw: { coal: [], gas_ccgt: [], gas_ct: [], oil: [], total: [] },
        dispatch_twh: { coal: [], gas_ccgt: [], gas_ct: [], oil: [], total: [] },
        capacity_factor_pct: { coal: [], gas_ccgt: [], gas_ct: [], oil: [] },
    };

    for (let i = 0; i < FLEET_THRESHOLDS.length; i++) {
        let npC = 0, npG = 0, npT = 0, npO = 0;
        let dC = 0, dG = 0, dT = 0, dO = 0;
        ISO_LIST.forEach(iso => {
            const m = isoData[iso];
            if (!m) return;
            npC += m.nameplate_gw.coal[i] || 0;
            npG += m.nameplate_gw.gas_ccgt[i] || 0;
            npT += m.nameplate_gw.gas_ct[i] || 0;
            npO += m.nameplate_gw.oil[i] || 0;
            dC += m.dispatch_twh.coal[i] || 0;
            dG += m.dispatch_twh.gas_ccgt[i] || 0;
            dT += m.dispatch_twh.gas_ct[i] || 0;
            dO += m.dispatch_twh.oil[i] || 0;
        });
        agg.nameplate_gw.coal.push(+npC.toFixed(2));
        agg.nameplate_gw.gas_ccgt.push(+npG.toFixed(2));
        agg.nameplate_gw.gas_ct.push(+npT.toFixed(2));
        agg.nameplate_gw.oil.push(+npO.toFixed(2));
        agg.nameplate_gw.total.push(+(npC + npG + npT + npO).toFixed(2));
        agg.dispatch_twh.coal.push(+dC.toFixed(1));
        agg.dispatch_twh.gas_ccgt.push(+dG.toFixed(1));
        agg.dispatch_twh.gas_ct.push(+dT.toFixed(1));
        agg.dispatch_twh.oil.push(+dO.toFixed(1));
        agg.dispatch_twh.total.push(+(dC + dG + dT + dO).toFixed(1));

        const cfPct = (twh, gw) => gw > 0 ? +((twh * 1000) / (gw * 8760) * 100).toFixed(1) : 0;
        agg.capacity_factor_pct.coal.push(cfPct(dC, npC));
        agg.capacity_factor_pct.gas_ccgt.push(cfPct(dG, npG));
        agg.capacity_factor_pct.gas_ct.push(cfPct(dT, npT));
        agg.capacity_factor_pct.oil.push(cfPct(dO, npO));
    }
    return agg;
}

/**
 * Compute company-specific nameplate capacity, dispatch, and capacity factor
 * at each SBTi threshold. Unlike computeFleetMixOverTime (which scales capacity
 * by dispatch fraction), this separates physical nameplate from economic dispatch:
 *   - Coal/oil nameplate: binary - full capacity until regional dispatch → 0, then retired
 *   - Gas nameplate: ALWAYS full (plants stay on grid as capacity resources)
 *   - Clean nameplate: ALWAYS full (unaffected by fossil dispatch decline)
 *   - Dispatch: merit-order model (same as computeFleetMixOverTime)
 *   - CF: dispatch / (nameplate × 8760h)
 */
function computeCompanyCapacityDispatch(company) {
    const allFuels = [...new Set(company.plants.map(p => p.fuel))];
    const out = {
        thresholds: [], years: [], fuels: allFuels,
        nameplate_gw: {}, dispatch_twh: {}, capacity_factor_pct: {},
    };
    allFuels.forEach(f => {
        out.nameplate_gw[f] = [];
        out.dispatch_twh[f] = [];
        out.capacity_factor_pct[f] = [];
    });
    out.nameplate_gw.total = [];
    out.dispatch_twh.total = [];

    // Also track by ISO for regional breakdown over time
    const isos = [...new Set(company.plants.map(p => p.iso))];
    out.isos = isos;
    out.nameplate_by_iso = {};
    out.dispatch_by_iso = {};
    // Per-ISO per-fuel breakdowns for ISO drill-down
    out.nameplate_by_iso_fuel = {};
    out.dispatch_by_iso_fuel = {};
    out.cf_by_iso_fuel = {};
    isos.forEach(iso => {
        out.nameplate_by_iso[iso] = [];
        out.dispatch_by_iso[iso] = [];
        out.nameplate_by_iso_fuel[iso] = {};
        out.dispatch_by_iso_fuel[iso] = {};
        out.cf_by_iso_fuel[iso] = {};
        allFuels.forEach(f => {
            out.nameplate_by_iso_fuel[iso][f] = [];
            out.dispatch_by_iso_fuel[iso][f] = [];
            out.cf_by_iso_fuel[iso][f] = [];
        });
    });

    FLEET_THRESHOLDS.forEach(t => {
        const year = FLEET_THRESHOLD_YEARS[t];
        out.thresholds.push(t);
        out.years.push(year);

        let totalNp = 0, totalDisp = 0;
        const isoNp = {}, isoDisp = {};
        isos.forEach(iso => { isoNp[iso] = 0; isoDisp[iso] = 0; });
        const fuelNpAcc = {}, fuelDispAcc = {};
        allFuels.forEach(f => { fuelNpAcc[f] = 0; fuelDispAcc[f] = 0; });
        // Per-ISO per-fuel accumulators
        const isoFuelNp = {}, isoFuelDisp = {};
        isos.forEach(iso => {
            isoFuelNp[iso] = {};
            isoFuelDisp[iso] = {};
            allFuels.forEach(f => { isoFuelNp[iso][f] = 0; isoFuelDisp[iso][f] = 0; });
        });

        company.plants.forEach(plant => {
            const iso = plant.iso;
            const decline = FOSSIL_DISPATCH_DECLINE[iso];

            if (CLEAN_FUELS.includes(plant.fuel)) {
                fuelNpAcc[plant.fuel] += plant.capacity_mw;
                fuelDispAcc[plant.fuel] += plant.generation_twh;
                isoNp[iso] = (isoNp[iso] || 0) + plant.capacity_mw;
                isoDisp[iso] = (isoDisp[iso] || 0) + plant.generation_twh;
                if (isoFuelNp[iso]) { isoFuelNp[iso][plant.fuel] += plant.capacity_mw; isoFuelDisp[iso][plant.fuel] += plant.generation_twh; }
                return;
            }

            if (!decline) {
                fuelNpAcc[plant.fuel] += plant.capacity_mw;
                fuelDispAcc[plant.fuel] += plant.generation_twh;
                isoNp[iso] = (isoNp[iso] || 0) + plant.capacity_mw;
                isoDisp[iso] = (isoDisp[iso] || 0) + plant.generation_twh;
                if (isoFuelNp[iso]) { isoFuelNp[iso][plant.fuel] += plant.capacity_mw; isoFuelDisp[iso][plant.fuel] += plant.generation_twh; }
                return;
            }

            const tData = decline.thresholds[String(t)];
            if (!tData) return;
            const remaining = tData.p50;

            if (plant.fuel === 'coal') {
                const coalFrac = decline.coal_cap_twh > 0 ?
                    Math.min(1, remaining.coal / decline.coal_cap_twh) : 0;
                // Nameplate: binary - running or retired
                const np = remaining.coal > 0.01 ? plant.capacity_mw : 0;
                const disp = plant.generation_twh * coalFrac;
                fuelNpAcc[plant.fuel] += np;
                fuelDispAcc[plant.fuel] += disp;
                isoNp[iso] = (isoNp[iso] || 0) + np;
                isoDisp[iso] = (isoDisp[iso] || 0) + disp;
                if (isoFuelNp[iso]) { isoFuelNp[iso][plant.fuel] += np; isoFuelDisp[iso][plant.fuel] += disp; }
            } else if (plant.fuel === 'oil') {
                const oilFrac = decline.oil_cap_twh > 0 ?
                    Math.min(1, (remaining.oil || 0) / decline.oil_cap_twh) : 0;
                const np = (remaining.oil || 0) > 0.01 ? plant.capacity_mw : 0;
                const disp = plant.generation_twh * oilFrac;
                fuelNpAcc[plant.fuel] += np;
                fuelDispAcc[plant.fuel] += disp;
                isoNp[iso] = (isoNp[iso] || 0) + np;
                isoDisp[iso] = (isoDisp[iso] || 0) + disp;
                if (isoFuelNp[iso]) { isoFuelNp[iso][plant.fuel] += np; isoFuelDisp[iso][plant.fuel] += disp; }
            } else if (plant.fuel === 'gas_ccgt' || plant.fuel === 'gas_peaker') {
                // Gas: nameplate ALWAYS stays (no physical retirement)
                fuelNpAcc[plant.fuel] += plant.capacity_mw;
                isoNp[iso] = (isoNp[iso] || 0) + plant.capacity_mw;
                if (isoFuelNp[iso]) { isoFuelNp[iso][plant.fuel] += plant.capacity_mw; }

                // Dispatch: merit-order (same logic as computeFleetMixOverTime)
                const gas_remaining = remaining.gas;
                const iso_ccgt_gen = company.plants
                    .filter(p => p.iso === iso && p.fuel === 'gas_ccgt')
                    .reduce((s, p) => s + p.generation_twh, 0);
                const iso_peaker_gen = company.plants
                    .filter(p => p.iso === iso && p.fuel === 'gas_peaker')
                    .reduce((s, p) => s + p.generation_twh, 0);
                const iso_total_gas = iso_ccgt_gen + iso_peaker_gen;
                const bfp = (100 - decline.baseline_clean_pct) / 100;
                const total_regional_gas = decline.base_demand_twh * bfp -
                    decline.coal_cap_twh - (decline.oil_cap_twh || 0);
                const rgf = total_regional_gas > 0 ?
                    Math.min(1, gas_remaining / total_regional_gas) : 0;

                let frac;
                if (plant.fuel === 'gas_ccgt') {
                    const ps = iso_total_gas > 0 ? iso_peaker_gen / iso_total_gas : 0;
                    frac = ps < 1 ? Math.min(1, rgf / (1 - ps)) : rgf;
                } else {
                    const cs = iso_total_gas > 0 ? iso_ccgt_gen / iso_total_gas : 0;
                    const ac = Math.max(0, rgf - cs);
                    frac = cs < 1 ? Math.min(1, ac / (1 - cs)) : 0;
                }
                const disp = plant.generation_twh * Math.min(1, frac);
                fuelDispAcc[plant.fuel] += disp;
                isoDisp[iso] = (isoDisp[iso] || 0) + disp;
                if (isoFuelDisp[iso]) { isoFuelDisp[iso][plant.fuel] += disp; }
            }
        });

        // Store results
        allFuels.forEach(f => {
            const np = fuelNpAcc[f];
            const disp = fuelDispAcc[f];
            out.nameplate_gw[f].push(+(np / 1000).toFixed(3));
            out.dispatch_twh[f].push(+disp.toFixed(2));
            const cf = np > 0 ? +((disp * 1e6) / (np * 8760) * 100).toFixed(1) : 0;
            out.capacity_factor_pct[f].push(cf);
            totalNp += np;
            totalDisp += disp;
        });
        out.nameplate_gw.total.push(+(totalNp / 1000).toFixed(3));
        out.dispatch_twh.total.push(+totalDisp.toFixed(2));

        isos.forEach(iso => {
            out.nameplate_by_iso[iso].push(+((isoNp[iso] || 0) / 1000).toFixed(3));
            out.dispatch_by_iso[iso].push(+((isoDisp[iso] || 0)).toFixed(2));
            // Per-ISO per-fuel
            allFuels.forEach(f => {
                const np = (isoFuelNp[iso] && isoFuelNp[iso][f]) || 0;
                const disp = (isoFuelDisp[iso] && isoFuelDisp[iso][f]) || 0;
                out.nameplate_by_iso_fuel[iso][f].push(+(np / 1000).toFixed(3));
                out.dispatch_by_iso_fuel[iso][f].push(+disp.toFixed(2));
                const cf = np > 0 ? +((disp * 1e6) / (np * 8760) * 100).toFixed(1) : 0;
                out.cf_by_iso_fuel[iso][f].push(cf);
            });
        });
    });

    return out;
}
