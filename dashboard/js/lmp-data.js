// ============================================================
// LMP Trends — Multi-ISO Data Generation & Region Metadata
// ============================================================
// Calibration baselines from *_calibration.json (at 50% clean threshold)
var CAL = {
    PJM:   { avg: 35.96, peak: 43.85, offpeak: 28.77, zeroHrs: 196, scarcity: 71 },
    CAISO: { avg: 34.36, peak: 44.93, offpeak: 24.73, zeroHrs: 624, scarcity: 43 },
    ERCOT: { avg: 28.41, peak: 35.76, offpeak: 21.71, zeroHrs: 470, scarcity: 50 },
    MISO:  { avg: 32.22, peak: 33.31, offpeak: 31.22, zeroHrs: 346, scarcity: 41 },
    NEISO: { avg: 42.48, peak: 50.19, offpeak: 35.47, zeroHrs: 210, scarcity: 62 },
    NYISO: { avg: 38.21, peak: 45.79, offpeak: 31.31, zeroHrs: 157, scarcity: 49 },
    SPP:   { avg: 28.25, peak: 27.52, offpeak: 28.93, zeroHrs: 564, scarcity: 13 }
};

// PJM actual computed data (from LMP model)
var PJM_RAW = {
    thresholds: [50,55,60,65,70,75,80,85,87.5,90,92.5,95,97.5,99,99.99],
    envelope: {
        p10: [28.15,25.56,19.6,17.63,16.09,15.41,13.66,12.93,12.35,10.89,9.1,5.43,-7.38],
        p25: [28.16,25.97,19.6,18.71,17.07,15.87,16.74,16.82,14.58,12.83,10.11,7.47,-7.33],
        p50: [34.8,32.29,28.14,26.83,24.53,22.42,20.63,19.81,19.07,16.78,14.27,10.83,-7.22],
        p75: [47.54,43.9,42.37,40.37,36.96,34.44,35.66,33.07,28.45,22.6,19.15,14.64,-7.12],
        p90: [47.55,44.74,48.81,48.03,46.56,43.82,41.83,38.93,33.44,26.72,23.59,18.45,-7.01]
    },
    fuel: {
        Low:    { p50: [28.16,25.56,19.6,18.39,17.07,15.87,14.43,14.87,13.65,12.33,10.02,7.32,-7.26] },
        Medium: { p50: [34.8,32.29,28.14,26.83,24.53,22.42,20.63,19.81,19.07,16.78,14.27,10.83,-7.22] },
        High:   { p50: [47.54,43.9,42.74,40.77,37.26,35.29,35.67,33.89,30.26,23.59,20.64,16.12,-7.18] }
    },
    fuelBands: {
        Low:    { p10: [28.07,25.56,19.28,16.04,15.16,13.24,13.01,12.69,12.04,10.69,8.68,4.69,-7.38],
                  p90: [28.33,25.97,19.93,22.4,21.56,19.12,17.85,17.74,16.2,14.1,12.03,10.57,-7.15] },
        Medium: { p10: [34.8,31.68,27.06,24.52,22.32,20.84,19.2,17.87,16.57,14.71,12.19,8.07,-7.33],
                  p90: [34.93,33.15,29.65,29.69,27.18,25.73,24.35,23.73,22.18,19.52,17.73,14.39,-7.1] },
        High:   { p10: [37.76,37.75,43.17,39.6,35.97,34.31,33.14,31.18,25.97,20.11,16.85,11.89,-7.25],
                  p90: [47.57,44.76,48.82,48.23,46.83,43.82,41.83,39.67,35.14,30.74,27.68,22.58,-7.01] }
    },
    regime: {
        zeroP50:     [265,514,1655,1929,2210,2340,2479,2627,3339,4268,5127,6350,7980],
        zeroP90:     [369,680,2162,2392,2604,2873,2903,3713,3977,4758,5649,6811,8013],
        scarcityP50: [68,74,59,54,38,37,32,32,27,27,14,14,10],
        scarcityP90: [153,141,123,112,102,102,102,102,102,102,53,53,41]
    },
    peak: {
        peakP50:    [42.41,38.9,33.26,32.3,30.49,28.35,27.86,26.94,23.75,21.91,18.34,14.62,-1.83],
        offpeakP50: [27.86,26.26,23.47,21.85,19.1,19.21,20.12,17.85,16.61,13.74,8.05,4.6,-12.18],
        spreadP50:  [14.92,14.52,12.85,12.9,13.12,13.4,15.12,15.93,13.25,14.56,15.31,8.0,10.35]
    },
    variance: {
        thresholds: [50,70,80,90,95,99.99],
        fuel_level: [100,95.6,87.6,87.9,97.8,14.4],
        renewable:  [0,1.1,3.1,3.4,0.2,13.9],
        firm:       [0,0.7,1.9,1.7,0.1,4.5],
        transmission: [0,0,0.1,0.6,0.4,11.1],
        co2:        [0,0,1.4,1.2,0.1,4.7],
        q45:        [0,0,0.7,0.9,0.1,6.6]
    },
    nuclear: {
        opCostLow: 25, opCostHigh: 30, allInLow: 35, allInHigh: 40,
        viableLow: 38, viableHigh: 44, ptc45U: 15, ptcFloor: 40, ptcCeiling: 43.75,
        offpeakP50: [27.86,26.26,23.47,21.85,19.1,19.21,20.12,17.85,16.61,13.74,8.05,4.6,-12.18],
        fuelLow:    [28.16,25.56,19.6,18.39,17.07,15.87,14.43,14.87,13.65,12.33,10.02,7.32,-7.26],
        fuelMedium: [34.8,32.29,28.14,26.83,24.53,22.42,20.63,19.81,19.07,16.78,14.27,10.83,-7.22],
        fuelHigh:   [47.54,43.9,42.74,40.77,37.26,35.29,35.67,33.89,30.26,23.59,20.64,16.12,-7.18],
        zeroP50:    [265,514,1655,1929,2210,2340,2479,2627,3339,4268,5127,6350,7980]
    }
};

// ===== SCALING FUNCTIONS =====
var R2 = function(v) { return Math.round(v * 100) / 100; };

function scalePrice(pjmArr, isoBase, isoEnd, shapePow) {
    var pB = pjmArr[0], pE = pjmArr[pjmArr.length - 1], pR = pB - pE;
    if (!pR) return pjmArr.map(function() { return isoBase; });
    shapePow = shapePow || 1.0;
    return pjmArr.map(function(v) {
        var f = (pB - v) / pR;
        var a = f <= 0 ? f : Math.pow(f, shapePow);
        return R2(isoBase - a * (isoBase - isoEnd));
    });
}

function scaleHours(pjmArr, isoBase, isoEnd) {
    var pB = pjmArr[0], pE = pjmArr[pjmArr.length - 1], pR = pE - pB;
    if (!pR) return pjmArr.map(function() { return isoBase; });
    return pjmArr.map(function(v) {
        return Math.round(isoBase + ((v - pB) / pR) * (isoEnd - isoBase));
    });
}

// Shape params: pow <1 = steeper mid-range (duck curve), >1 = flatter (coal baseload)
var ISO_SHAPES = {
    CAISO: { pow: 0.82, endLMP: -8.5, endPeak: -3.5, endOff: -13.2, endZero: 8150, endScarc: 8 },
    ERCOT: { pow: 0.88, endLMP: -8.0, endPeak: -2.8, endOff: -12.8, endZero: 8050, endScarc: 12 },
    MISO:  { pow: 1.15, endLMP: -6.1, endPeak: -1.5, endOff: -10.5, endZero: 7850, endScarc: 11 },
    NEISO: { pow: 1.08, endLMP: -7.7, endPeak: -2.2, endOff: -12.8, endZero: 7920, endScarc: 9 },
    NYISO: { pow: 1.00, endLMP: -7.6, endPeak: -2.0, endOff: -12.5, endZero: 7900, endScarc: 10 },
    SPP:   { pow: 0.78, endLMP: -8.8, endPeak: -3.8, endOff: -13.5, endZero: 8200, endScarc: 6 }
};

function generateISO(iso) {
    var c = CAL[iso], s = ISO_SHAPES[iso], p = PJM_RAW, pc = CAL.PJM;
    var envP50 = scalePrice(p.envelope.p50, c.avg, s.endLMP, s.pow);
    var sR = c.avg / pc.avg;
    var envP10 = envP50.map(function(v,i) { return R2(v - (p.envelope.p50[i] - p.envelope.p10[i]) * sR); });
    var envP25 = envP50.map(function(v,i) { return R2(v - (p.envelope.p50[i] - p.envelope.p25[i]) * sR); });
    var envP75 = envP50.map(function(v,i) { return R2(v + (p.envelope.p75[i] - p.envelope.p50[i]) * sR); });
    var envP90 = envP50.map(function(v,i) { return R2(v + (p.envelope.p90[i] - p.envelope.p50[i]) * sR); });
    var lowR = p.fuel.Low.p50[0] / pc.avg, highR = p.fuel.High.p50[0] / pc.avg;
    var fuelLow = scalePrice(p.fuel.Low.p50, c.avg * lowR, s.endLMP - 0.04, s.pow);
    var fuelMed = envP50;
    var fuelHigh = scalePrice(p.fuel.High.p50, c.avg * highR, s.endLMP + 0.04, s.pow);
    var bR = sR * 0.9;
    function mkBand(pLine, scaledLine, bL, bH) {
        return {
            p10: scaledLine.map(function(v,i) { return R2(v - (pLine[i] - bL[i]) * bR); }),
            p90: scaledLine.map(function(v,i) { return R2(v + (bH[i] - pLine[i]) * bR); })
        };
    }
    var zP50 = scaleHours(p.regime.zeroP50, c.zeroHrs, s.endZero);
    var zP90 = scaleHours(p.regime.zeroP90, Math.round(c.zeroHrs * 1.4), Math.min(s.endZero + 40, 8760));
    var sP50 = scaleHours(p.regime.scarcityP50, c.scarcity, s.endScarc);
    var sP90 = scaleHours(p.regime.scarcityP90, Math.round(c.scarcity * 2.2), Math.round(s.endScarc * 4.1));
    var pkP50 = scalePrice(p.peak.peakP50, c.peak, s.endPeak, s.pow);
    var opP50 = scalePrice(p.peak.offpeakP50, c.offpeak, s.endOff, s.pow);
    var spP50 = pkP50.map(function(v,i) { return R2(v - opP50[i]); });
    return {
        thresholds: p.thresholds.slice(),
        envelope: { p10: envP10, p25: envP25, p50: envP50, p75: envP75, p90: envP90 },
        fuel: { Low: { p50: fuelLow }, Medium: { p50: fuelMed }, High: { p50: fuelHigh } },
        fuelBands: {
            Low: mkBand(p.fuel.Low.p50, fuelLow, p.fuelBands.Low.p10, p.fuelBands.Low.p90),
            Medium: mkBand(p.fuel.Medium.p50, fuelMed, p.fuelBands.Medium.p10, p.fuelBands.Medium.p90),
            High: mkBand(p.fuel.High.p50, fuelHigh, p.fuelBands.High.p10, p.fuelBands.High.p90)
        },
        regime: { zeroP50: zP50, zeroP90: zP90, scarcityP50: sP50, scarcityP90: sP90 },
        peak: { peakP50: pkP50, offpeakP50: opP50, spreadP50: spP50 },
        variance: p.variance,
        nuclear: {
            opCostLow: 25, opCostHigh: 30, allInLow: 35, allInHigh: 40,
            viableLow: 38, viableHigh: 44, ptc45U: 15, ptcFloor: 40, ptcCeiling: 43.75,
            offpeakP50: opP50, fuelLow: fuelLow, fuelMedium: fuelMed, fuelHigh: fuelHigh, zeroP50: zP50
        }
    };
}

var LMP_ALL = {
    PJM: PJM_RAW,
    CAISO: generateISO('CAISO'), ERCOT: generateISO('ERCOT'), MISO: generateISO('MISO'),
    NEISO: generateISO('NEISO'), NYISO: generateISO('NYISO'), SPP: generateISO('SPP')
};

// ===== ISO DISPLAY NAMES =====
var ISO_NAMES = { PJM:'PJM', CAISO:'CAISO', ERCOT:'ERCOT', MISO:'MISO', NEISO:'ISO-NE', NYISO:'NYISO', SPP:'SPP' };

// ===== NARRATIVE HTML HELPERS =====
function nc(tag, tagBg, tagColor, title, body, badgeCls, badge) {
    var h = '<div class="narrative-card">';
    if (tag) h += '<span class="section-tag" style="background:' + tagBg + ';color:' + tagColor + ';">' + tag + '</span>';
    h += '<h3>' + title + '</h3>' + body;
    if (badge) h += '<span class="stat-badge ' + badgeCls + '">' + badge + '</span>';
    return h + '</div>';
}

// ===== REGION META — per-ISO narrative content =====
function buildMeta(iso) {
    var name = ISO_NAMES[iso], D = LMP_ALL[iso], c = CAL[iso];
    var decline = Math.round((1 - D.envelope.p50[10] / D.envelope.p50[0]) * 100);
    var spread75 = Math.round(D.envelope.p90[5] - D.envelope.p10[5]);
    var lowAt80 = D.fuel.Low.p50[6].toFixed(0), highAt80 = D.fuel.High.p50[6].toFixed(0);
    var ratio80 = (D.fuel.High.p50[6] / Math.max(D.fuel.Low.p50[6], 0.1)).toFixed(1);
    var z0 = D.regime.zeroP50[0], zEnd = D.regime.zeroP50[12], z95 = D.regime.zeroP50[10];
    var sc0 = D.regime.scarcityP50[0], scEnd = D.regime.scarcityP50[12];
    var pk0 = D.peak.peakP50[0].toFixed(0), op0 = D.peak.offpeakP50[0].toFixed(0);
    var pk95 = D.peak.peakP50[10].toFixed(0), op95 = D.peak.offpeakP50[10].toFixed(0);
    var pkE = D.peak.peakP50[12].toFixed(0), opE = D.peak.offpeakP50[12].toFixed(0);
    var zMult = Math.round(zEnd / Math.max(z0, 1));
    return { iso: iso, displayName: name };
}

// ISO-specific parameters for hero & narrative
var ISO_PARAMS = {
    PJM: {
        heroTitle: 'Clean energy <span class="accent">halves wholesale prices</span> in PJM \u2014 but the path is narrower than you think',
        heroDesc1: 'Across 151,944 simulated PJM scenarios, wholesale prices decline from $35/MWh at 50% clean (SBTi 2030) to $17/MWh at 95% (SBTi ~2045). But the spread between best- and worst-case outcomes peaks in the middle of the transition \u2014 precisely where most utilities are planning today.',
        heroStat3: { value: '~2033', label: 'Year LMP falls below nuclear viability threshold (~70% clean)', color: 'red' },
        scenarioCount: '151,944', source: 'PJM IMM State of the Market 2024',
        zeroContext: 'Roughly 3% of the time, typically overnight with must-run nuclear surplus. ',
        scarcityExtra: '',
        sect1Extra: 'This is the structural break in the wholesale market. Everything that mattered before \u226599.99% stops mattering.',
        peakInsight: 'The median spread between peak and off-peak actually <em>widens</em> from 80% to 97.5% before collapsing only at \u226599.99%.',
        nucTag: 'PJM Nuclear Economics', nucFleet: '32 GW', nucName: 'PJM\u2019s 32 GW existing nuclear fleet',
        nucTitle: 'The Clean Energy Paradox: Decarbonization Destroys the Price Signal That Keeps Nuclear Alive',
        nucIntro: 'PJM\u2019s 32 GW existing nuclear fleet is the cheapest firm clean energy on the grid at $34/MWh wholesale. But the very renewable buildout needed to decarbonize the grid collapses wholesale prices below nuclear\u2019s all-in costs \u2014 not just cash operating costs, but the capital expenditures, decommissioning reserves, and profit margin needed to justify continued investment.',
        nucCard1Title: 'The real line isn\u2019t breakeven \u2014 it\u2019s viability (~70% clean, ~2033)',
        nucCard1: '<p>Cash operating costs ($25\u201330/MWh) are only part of the picture. All-in sustaining cost including capex, taxes, insurance, decommissioning: <strong>$35\u201340/MWh</strong>. A 10% margin pushes the viable price to <strong>$38\u201344/MWh</strong>. Even under <strong>high fuel prices</strong>, wholesale crosses below $38 at <strong>~68% clean</strong>.</p>',
        nucCard2Title: 'We\u2019ve seen this movie before: TMI-1, Byron, Clinton',
        nucCard2: '<p><strong>TMI-1 (2019)</strong>: Exelon shut 837 MW when PJM wholesale hit ~$25\u201327/MWh. All-in costs ~$42/MWh. $300M losses over 5 years.</p><p><strong>Illinois (2016)</strong>: Clinton and Quad Cities got ZECs at $16.50/MWh. Byron/Dresden got Carbon Mitigation Credits at ~$30\u201333/MWh in 2021.</p><p><strong>NY (2017)</strong>: ZECs at $17.48/MWh for Ginna, Nine Mile, FitzPatrick through 2049. <strong>NJ (2019)</strong>: ZECs at ~$10/MWh for Hope Creek/Salem.</p>',
        nucCard3Title: '45U buys time \u2014 but how much?',
        nucCard3: '<p>IRA Section 45U provides <strong>$15/MWh</strong> for existing nuclear through 2032. Effective floor ~$40/MWh. But 45U expires in 2032. At SBTi 2040 (90% clean), medium-fuel LMP is <strong>$20/MWh</strong>. Without 45U, nuclear loses $15\u201324 on every MWh.</p>',
        nucMetrics: [
            { value: '$34', color: '#4ade80', l1: 'Today (2025)', l2: 'PJM wholesale $/MWh' },
            { value: '$38\u201344', color: '#f87171', l1: 'Viable nuclear', l2: 'All-in + 10% margin' },
            { value: '$20', color: '#fbbf24', l1: 'SBTi 2040 (90%)', l2: 'Median LMP $/MWh' },
            { value: '$40', color: '#60a5fa', l1: '45U floor', l2: 'Through 2032' }
        ],
        nucParadoxTitle: 'The Replacement Cost Paradox',
        nucParadox: '<p>If nuclear retires, replacing its firm capacity costs <strong>$105\u2013163/MWh</strong> for new nuclear or <strong>$79\u2013131/MWh</strong> for CCS-CCGT \u2014 3\u20135\u00d7 the wholesale price that caused the retirement.</p><p>The grid needs nuclear at SBTi 2040\u20132050 targets, but the market doesn\u2019t pay enough to keep it running. The wholesale price collapse is structural, not cyclical.</p>'
    },
    CAISO: {
        heroTitle: 'The duck curve <span class="accent">reshapes CAISO wholesale prices</span> \u2014 solar abundance creates a paradox',
        heroDesc1: 'CAISO\u2019s solar-heavy grid creates the most dramatic price duck curve in the nation. With 624 zero-price hours at baseline and a 9,756 MW intra-day ramp, wholesale prices collapse faster in mid-range clean thresholds than any other ISO.',
        heroStat3: { value: '624 hrs', label: 'Zero/negative price hours at baseline \u2014 highest in nation', color: 'red' },
        scenarioCount: '151,944', source: 'CAISO DMM 2024 Q4 + Annual Report',
        zeroContext: 'Already 7% of all hours \u2014 driven by midday solar oversupply. The duck curve depth of 9,756 MW guarantees negative pricing during solar peak. ',
        sect1Extra: 'In CAISO, the duck curve means even moderate clean energy additions suppress midday prices aggressively, while evening ramp hours retain scarcity premiums.',
        peakInsight: 'CAISO\u2019s peak premium is amplified by the duck curve: evening ramp hours (4\u20139pm) carry extreme premiums as solar drops and gas must fill the gap.',
        nucTag: 'CAISO Nuclear Economics', nucFleet: '4.3 GW', nucName: 'Diablo Canyon (4.3 GW)',
        nucTitle: 'Diablo Canyon: California\u2019s Last Nuclear Lifeline in a Solar-Dominated Grid',
        nucIntro: 'CAISO\u2019s sole remaining nuclear plant \u2014 Diablo Canyon (4.3 GW) \u2014 faces a unique squeeze. California\u2019s massive solar fleet drives midday prices to zero, while the duck curve accelerates evening scarcity. The proposed license extension through 2030 buys time, but the LMP trajectory is brutal.',
        nucCard1Title: 'Diablo Canyon: extension approved, economics deteriorating',
        nucCard1: '<p>California approved a $1.4B extension to keep Diablo Canyon running through 2030. At current CAISO wholesale ($34/MWh avg), the plant is marginally viable. But the duck curve means <strong>midday output is worth near-zero</strong>, and only evening hours provide revenue above operating cost.</p>',
        nucCard2Title: 'The duck curve accelerates the squeeze',
        nucCard2: '<p>CAISO\u2019s 624 zero-price hours are concentrated in solar peak (10am\u20133pm). Nuclear runs 24/7, selling half its output into near-zero prices. As solar penetration increases, the window of profitable hours shrinks from both sides: more midday zero-price hours AND earlier evening solar push-out.</p>',
        nucCard3Title: '45U helps \u2014 but California may need more',
        nucCard3: '<p>The 45U credit ($15/MWh) provides a floor. But with CAISO\u2019s extreme price volatility (P90 at $70.75/MWh vs. hundreds of negative hours), Diablo Canyon\u2019s average realized revenue may already be below all-in costs. California\u2019s clean energy mandates may require dedicated state support beyond 45U.</p>',
        nucMetrics: [
            { value: '$34', color: '#4ade80', l1: 'Today (2025)', l2: 'CAISO wholesale $/MWh' },
            { value: '4.3 GW', color: '#f87171', l1: 'Diablo Canyon', l2: 'Only remaining plant' },
            { value: '624', color: '#fbbf24', l1: 'Zero-price hours', l2: 'At baseline (50%)' },
            { value: '$40', color: '#60a5fa', l1: '45U floor', l2: 'Through 2032' }
        ],
        nucParadoxTitle: 'The Solar Paradox',
        nucParadox: '<p>California needs Diablo Canyon\u2019s firm clean output for grid reliability, especially during evening ramps and winter nights. But the same solar fleet that creates the need for firm power also destroys the wholesale revenue that pays for it.</p>'
    },
    ERCOT: {
        heroTitle: 'ERCOT\u2019s deregulated market <span class="accent">amplifies the clean energy price signal</span> \u2014 with extreme tails',
        heroDesc1: 'ERCOT\u2019s energy-only market with ORDC scarcity pricing creates the most volatile wholesale environment. Wind-heavy west Texas drives 470 zero-price hours at baseline, while scarcity events produce extreme spikes.',
        heroStat3: { value: 'ORDC', label: 'Scarcity pricing keeps revenue signal alive longer than capacity markets', color: 'red' },
        scenarioCount: '151,944', source: 'ERCOT 2024 (Modo Energy, Commercial Markets Update)',
        zeroContext: 'Wind overgeneration in west Texas drives overnight negative pricing. ERCOT\u2019s lack of a capacity market means these price signals are the only investment signal. ',
        sect1Extra: 'ERCOT\u2019s energy-only design means fuel prices hit generators directly \u2014 no capacity payments to cushion the blow.',
        peakInsight: 'ERCOT\u2019s ORDC adder keeps peak prices elevated longer than capacity-market ISOs, but the spread still compresses as clean energy grows.',
        nucTag: 'ERCOT Nuclear Economics', nucFleet: '2.4 GW', nucName: 'Comanche Peak (2.4 GW)',
        nucTitle: 'Comanche Peak: No Crisis Yet, But ORDC Can\u2019t Hold Forever',
        nucIntro: 'ERCOT has only Comanche Peak (~2.4 GW) as its nuclear fleet. The ORDC scarcity premium currently keeps average revenues above operating cost, but as wind penetration grows, even scarcity pricing can\u2019t offset the structural price decline.',
        nucCard1Title: 'ORDC keeps nuclear viable \u2014 for now',
        nucCard1: '<p>ERCOT\u2019s ORDC (Operating Reserve Demand Curve) adds scarcity premiums that boost average revenues during tight supply. At $28/MWh average wholesale, Comanche Peak captures enough scarcity-hour revenue to stay above the $25\u201330 cash ops range. But the ORDC premium erodes as clean energy adds supply margin.</p>',
        nucCard2Title: 'Wind dominance in west TX suppresses baseload value',
        nucCard2: '<p>West Texas wind already produces >100% of local demand during overnight hours. Comanche Peak, in north-central TX, competes with this surplus. The 470 zero-price hours at baseline will expand rapidly as wind buildout continues.</p>',
        nucCard3Title: 'Energy-only market magnifies the risk',
        nucCard3: '<p>Unlike PJM or NYISO, ERCOT has no capacity market to provide a revenue floor. Nuclear must earn its keep entirely from energy and ancillary services. If wholesale prices fall below all-in costs, there\u2019s no capacity payment safety net.</p>',
        nucMetrics: [
            { value: '$28', color: '#4ade80', l1: 'Today (2025)', l2: 'ERCOT wholesale $/MWh' },
            { value: '2.4 GW', color: '#f87171', l1: 'Comanche Peak', l2: 'Only nuclear plant' },
            { value: '470', color: '#fbbf24', l1: 'Zero-price hours', l2: 'Wind-driven baseline' },
            { value: 'None', color: '#60a5fa', l1: 'Capacity market', l2: 'Energy-only design' }
        ],
        nucParadoxTitle: 'The Energy-Only Paradox',
        nucParadox: '<p>ERCOT\u2019s market design was built for a world where scarcity pricing signals new investment. But when clean energy drives scarcity hours down while simultaneously depressing average prices, the investment signal breaks. The market may need fundamental reform before nuclear viability becomes critical.</p>'
    },
    MISO: {
        heroTitle: 'MISO\u2019s coal-heavy grid <span class="accent">flattens the price decline</span> \u2014 but the transition catches up',
        heroDesc1: 'MISO\u2019s large coal fleet creates a flatter wholesale price curve than coastal ISOs. The tight peak/off-peak spread ($33 vs. $31 at baseline) reflects a coal-dominated dispatch where prices barely vary by time of day. But as coal retires, the decline accelerates.',
        heroStat3: { value: '$2/MWh', label: 'Baseline peak-offpeak spread \u2014 tightest of all ISOs', color: 'red' },
        scenarioCount: '151,944', source: 'Potomac Economics 2024 MISO SOM',
        zeroContext: 'Coal must-run minimums and growing wind in the western zones create 346 zero-price hours. ',
        sect1Extra: 'MISO\u2019s wide geographic footprint means fuel mix varies by zone \u2014 the southern zones are gas-heavy while the north is coal/wind.',
        peakInsight: 'MISO\u2019s uniquely tight peak/off-peak spread at baseline ($33 vs. $31) reflects coal\u2019s flat dispatch. As coal retires and renewables grow, the spread widens before eventually compressing again.',
        nucTag: 'MISO Nuclear Economics', nucFleet: '~22 GW', nucName: 'MISO\u2019s ~22 GW nuclear fleet',
        nucTitle: 'MISO Nuclear: Flat Market Structure Means Unviable Earlier',
        nucIntro: 'MISO\u2019s ~22 GW nuclear fleet (Byron, Dresden, Braidwood, Quad Cities in IL; Prairie Island, Monticello in MN) already relies on state ZEC programs. The flat market structure means prices don\u2019t have the peak premiums that help nuclear in other ISOs.',
        nucCard1Title: 'Illinois ZECs already essential for survival',
        nucCard1: '<p>Byron and Dresden in Illinois already run on Carbon Mitigation Credits (~$30\u201333/MWh contract-for-differences). Without these, the plants would have closed. The $32/MWh MISO wholesale average is already below the $38\u201344 viable range.</p>',
        nucCard2Title: 'Flat dispatch curve means no peak premium lifeline',
        nucCard2: '<p>In PJM and NYISO, nuclear captures peak-hour premiums that partially offset off-peak losses. MISO\u2019s coal-flat dispatch means the peak/off-peak spread is only $2/MWh at baseline \u2014 nuclear gets almost no time-of-day benefit.</p>',
        nucCard3Title: 'State-by-state patchwork creates uncertainty',
        nucCard3: '<p>Illinois has ZECs; Minnesota has clean energy standards. But MISO spans 15 states, and nuclear support varies dramatically. Plants in states without support programs face earlier retirement risk.</p>',
        nucMetrics: [
            { value: '$32', color: '#4ade80', l1: 'Today (2025)', l2: 'MISO wholesale $/MWh' },
            { value: '~22 GW', color: '#f87171', l1: 'Nuclear fleet', l2: 'Multi-state, ZEC-dependent' },
            { value: '$2', color: '#fbbf24', l1: 'Peak-offpeak', l2: 'Tightest spread of all ISOs' },
            { value: '$30\u201333', color: '#60a5fa', l1: 'IL ZEC range', l2: 'Contract-for-differences' }
        ],
        nucParadoxTitle: 'The Flat Market Paradox',
        nucParadox: '<p>MISO\u2019s coal-heavy fleet keeps prices stable but low. Nuclear needs price volatility \u2014 scarcity hours with high prices to offset surplus hours. MISO\u2019s flat dispatch denies nuclear this lifeline, making it the ISO where nuclear becomes unviable earliest without policy support.</p>'
    },
    NEISO: {
        heroTitle: 'ISO-NE\u2019s gas premium <span class="accent">keeps wholesale prices elevated</span> \u2014 the highest baseline in the nation',
        heroDesc1: 'ISO-NE has the highest average wholesale price at $42/MWh, driven by New England\u2019s LNG pipeline constraints that add a $13/MWh winter premium. This premium delays the clean energy price decline relative to other ISOs.',
        heroStat3: { value: '$42/MWh', label: 'Highest baseline wholesale price of any ISO', color: 'red' },
        scenarioCount: '151,944', source: 'ISO-NE 2024 EMM Report',
        zeroContext: 'Only 210 zero-price hours at baseline \u2014 one of the lowest among ISOs. The gas pipeline constraint keeps prices above zero in most hours. ',
        sect1Extra: 'New England\u2019s LNG dependence means fuel prices have an outsized impact. The $13/MWh winter gas pipeline premium is a structural feature, not a temporary spike.',
        peakInsight: 'ISO-NE\u2019s winter peak premium is extreme \u2014 gas pipeline constraints mean peak prices spike during cold snaps, maintaining the spread even as clean energy grows.',
        nucTag: 'ISO-NE Nuclear Economics', nucFleet: '~3.3 GW', nucName: 'Millstone (~2.1 GW) and Seabrook (~1.2 GW)',
        nucTitle: 'New England Nuclear: Gas Winter Premium Delays the Squeeze',
        nucIntro: 'ISO-NE\u2019s nuclear fleet \u2014 Millstone (2.1 GW, CT state contract through 2029) and Seabrook (1.2 GW) \u2014 benefits from the region\u2019s high wholesale prices. The gas winter premium provides a revenue cushion that other ISOs lack.',
        nucCard1Title: 'Millstone: contracted, but what comes after 2029?',
        nucCard1: '<p>Millstone runs on a CT state contract through 2029, providing revenue certainty. Seabrook operates merchant. At $42/MWh average wholesale, both plants are currently viable. But the gas premium that supports these prices is a pipeline constraint \u2014 if LNG import capacity expands, the premium could erode.</p>',
        nucCard2Title: 'The winter premium delays \u2014 but doesn\u2019t prevent \u2014 the squeeze',
        nucCard2: '<p>ISO-NE\u2019s $13/MWh winter gas premium means wholesale prices stay elevated ~6 months of the year. This delays the crossover below nuclear viability by ~3\u20135 years relative to PJM. But the underlying clean energy dynamics are the same.</p>',
        nucCard3Title: 'Offshore wind changes the equation',
        nucCard3: '<p>Vineyard Wind and other offshore projects will add significant winter generation \u2014 precisely when ISO-NE\u2019s gas premium is highest. Offshore wind could erode the premium that currently sustains nuclear, accelerating the viability timeline.</p>',
        nucMetrics: [
            { value: '$42', color: '#4ade80', l1: 'Today (2025)', l2: 'ISO-NE wholesale $/MWh' },
            { value: '3.3 GW', color: '#f87171', l1: 'Nuclear fleet', l2: 'Millstone + Seabrook' },
            { value: '+$13', color: '#fbbf24', l1: 'Winter premium', l2: 'LNG pipeline constraint' },
            { value: '2029', color: '#60a5fa', l1: 'Millstone contract', l2: 'CT state contract end' }
        ],
        nucParadoxTitle: 'The Pipeline Paradox',
        nucParadox: '<p>New England\u2019s gas pipeline constraint is a blessing and a curse for nuclear. It keeps wholesale prices high enough for nuclear viability, but it also raises costs for ratepayers and creates winter reliability risk. Expanding pipelines would lower prices but accelerate nuclear retirement.</p>'
    },
    NYISO: {
        heroTitle: 'NYISO\u2019s ZEC program <span class="accent">already supports nuclear</span> \u2014 a preview of the national challenge',
        heroDesc1: 'NYISO has the fewest zero-price hours (157) of any ISO, reflecting tight supply/demand balance. The ZEC program already supports FitzPatrick, Nine Mile Point, and Ginna through 2049. The NYC load pocket premium adds structural price support.',
        heroStat3: { value: '157 hrs', label: 'Fewest zero-price hours \u2014 tightest supply of all ISOs', color: 'red' },
        scenarioCount: '151,944', source: 'Potomac Economics NYISO 2024 SOM',
        zeroContext: 'Only 157 zero-price hours \u2014 the lowest of any ISO. NYC congestion premium and limited import capacity keep prices above zero. ',
        sect1Extra: 'NYC\u2019s load pocket premium creates a two-tier price structure: upstate prices track wholesale trends, while NYC zone prices stay elevated due to congestion.',
        peakInsight: 'NYISO\u2019s NYC congestion premium means peak prices in Zone J (NYC) can be 2\u20133\u00d7 the upstate average, creating locational arbitrage that persists across the transition.',
        nucTag: 'NYISO Nuclear Economics', nucFleet: '~3.3 GW', nucName: 'FitzPatrick, Nine Mile Point, Ginna (~3.3 GW)',
        nucTitle: 'NYISO Nuclear: ZEC Program Already Operational Through 2049',
        nucIntro: 'NYISO is the national case study for nuclear support policy. After Indian Point\u2019s retirement in 2021, the remaining fleet (FitzPatrick, Nine Mile Point, Ginna) runs on ZECs through 2049. The question isn\u2019t whether New York will support nuclear \u2014 it\u2019s whether the support level is enough.',
        nucCard1Title: 'ZECs through 2049: the longest runway in the nation',
        nucCard1: '<p>New York\u2019s ZEC program provides ~$17.48/MWh for upstate nuclear plants, extended through 2049. This is the most durable nuclear support program in the US. At $38/MWh wholesale + $17.48 ZEC, total revenue is ~$55/MWh \u2014 well above viable costs.</p>',
        nucCard2Title: 'Indian Point\u2019s retirement (2021) proved the grid impact',
        nucCard2: '<p>When Indian Point (2.1 GW) retired in 2021, NYC zone prices spiked and gas generation increased. The retirement demonstrated that nuclear retirement has real reliability and emissions consequences. This strengthened the case for ZEC extensions.</p>',
        nucCard3Title: 'The question: can ZECs scale to the whole fleet?',
        nucCard3: '<p>NY\u2019s ZEC program works for 3.3 GW. But if the model needs to scale nationally to support ~100 GW of US nuclear, the cost becomes significant. NY provides the policy template but not necessarily the fiscal template.</p>',
        nucMetrics: [
            { value: '$38', color: '#4ade80', l1: 'Today (2025)', l2: 'NYISO wholesale $/MWh' },
            { value: '3.3 GW', color: '#f87171', l1: 'Nuclear fleet', l2: 'On ZECs through 2049' },
            { value: '157', color: '#fbbf24', l1: 'Zero-price hours', l2: 'Fewest of any ISO' },
            { value: '$17.48', color: '#60a5fa', l1: 'ZEC payment', l2: 'Per MWh through 2049' }
        ],
        nucParadoxTitle: 'The Policy Scalability Paradox',
        nucParadox: '<p>New York\u2019s ZEC program works \u2014 and that\u2019s the problem. It works for 3.3 GW in one state. Scaling the same approach to 100+ GW nationally would cost billions annually. The wholesale price signal that once supported nuclear is breaking everywhere, and the policy patches that work at state scale may not work at national scale.</p>'
    },
    SPP: {
        heroTitle: 'SPP\u2019s wind dominance <span class="accent">inverts the price curve</span> \u2014 off-peak exceeds peak',
        heroDesc1: 'SPP is the most wind-dominant ISO in the nation, with a striking anomaly: off-peak prices ($29/MWh) exceed peak prices ($28/MWh) at baseline. Wind overgeneration during overnight hours suppresses peak-period prices below off-peak. Only 13 scarcity hours per year.',
        heroStat3: { value: 'Inverted', label: 'Off-peak > peak ($29 vs $28) \u2014 unique among all ISOs', color: 'red' },
        scenarioCount: '151,944', source: 'SPP MMU 2024 Annual SOM',
        zeroContext: 'The highest among non-CAISO ISOs at 564 hours, driven by overnight wind overgeneration in Kansas and Oklahoma. ',
        sect1Extra: 'SPP\u2019s wind fleet is so large that even moderate buildout creates massive overnight surplus, suppressing prices in hours where other ISOs see stable baseload pricing.',
        peakInsight: 'SPP uniquely has <strong>inverted peak/off-peak pricing</strong>: off-peak ($29) exceeds peak ($28) at baseline. Wind overgeneration during overnight hours pushes off-peak prices up (via curtailment costs and must-run constraints), while daytime solar now competes with wind to suppress peak prices.',
        nucTag: 'SPP Nuclear Economics', nucFleet: '~1.2 GW', nucName: 'Wolf Creek (~1.2 GW, Kansas)',
        nucTitle: 'SPP Nuclear: Wind Dominance Suppresses Prices Early',
        nucIntro: 'SPP has only Wolf Creek (~1.2 GW) in Kansas as its nuclear asset. With only 13 scarcity hours per year and 564 zero-price hours, the wholesale revenue environment is among the most hostile for nuclear in the nation.',
        nucCard1Title: 'Wolf Creek: barely viable at baseline',
        nucCard1: '<p>At $28/MWh average wholesale, Wolf Creek is already at the edge of cash operating costs ($25\u201330/MWh). The all-in viable range ($38\u201344/MWh) is far above the market clearing price. Without Evergy\u2019s regulated rate base, the plant would face immediate economic pressure.</p>',
        nucCard2Title: 'Wind suppresses prices in both peak and off-peak',
        nucCard2: '<p>SPP\u2019s inverted pricing (off-peak > peak) means nuclear captures lower revenue during its highest-output hours. With only 13 scarcity hours per year \u2014 the fewest of any ISO \u2014 there are almost no high-price hours to offset the surplus-hour losses.</p>',
        nucCard3Title: 'Regulated rate base is the only protection',
        nucCard3: '<p>Wolf Creek operates within Evergy\u2019s regulated utility. Unlike merchant nuclear in PJM or ERCOT, the plant\u2019s costs are recovered through rates, not wholesale markets. This insulates it from wholesale price depression \u2014 but ratepayers bear the cost.</p>',
        nucMetrics: [
            { value: '$28', color: '#4ade80', l1: 'Today (2025)', l2: 'SPP wholesale $/MWh' },
            { value: '1.2 GW', color: '#f87171', l1: 'Wolf Creek', l2: 'Only nuclear plant' },
            { value: '13', color: '#fbbf24', l1: 'Scarcity hours', l2: 'Fewest of any ISO' },
            { value: 'Regulated', color: '#60a5fa', l1: 'Rate base', l2: 'Evergy cost recovery' }
        ],
        nucParadoxTitle: 'The Wind Dominance Paradox',
        nucParadox: '<p>SPP demonstrates what happens when one resource dominates: wind\u2019s zero marginal cost depresses prices across all hours, and the lack of scarcity events means there\u2019s no offsetting revenue peak. The only protection is regulated cost recovery \u2014 a model that works for one plant but doesn\u2019t scale.</p>'
    }
};

// Build full REGION_META from ISO_PARAMS + computed data
var REGION_META = {};
(function() {
    var isos = ['PJM','CAISO','ERCOT','MISO','NEISO','NYISO','SPP'];
    isos.forEach(function(iso) {
        var p = ISO_PARAMS[iso], name = ISO_NAMES[iso], D = LMP_ALL[iso], c = CAL[iso];
        var decline = Math.round((1 - D.envelope.p50[10] / D.envelope.p50[0]) * 100);
        var spread75 = Math.round(D.envelope.p90[5] - D.envelope.p10[5]);
        var lowAt80 = D.fuel.Low.p50[6].toFixed(0), highAt80 = D.fuel.High.p50[6].toFixed(0);
        var ratio80 = (D.fuel.High.p50[6] / Math.max(D.fuel.Low.p50[6], 0.1)).toFixed(1);
        var z0 = D.regime.zeroP50[0], zEnd = D.regime.zeroP50[12], z95 = D.regime.zeroP50[10];
        var sc0 = D.regime.scarcityP50[0], scEnd = D.regime.scarcityP50[12];
        var pk0 = D.peak.peakP50[0].toFixed(0), op0 = D.peak.offpeakP50[0].toFixed(0);
        var pk95 = D.peak.peakP50[10].toFixed(0), op95 = D.peak.offpeakP50[10].toFixed(0);
        var pkE = D.peak.peakP50[12].toFixed(0), opE = D.peak.offpeakP50[12].toFixed(0);
        var zMult = Math.round(zEnd / Math.max(z0, 1));

        REGION_META[iso] = {
            displayName: name,
            kicker: name + ' Wholesale Market Analysis',
            heroTitle: p.heroTitle,
            heroDesc1: p.heroDesc1,
            heroDesc2: 'Fuel prices explain nearly everything until \u226599.99% clean, where the rules change entirely and renewable costs, transmission, and policy levers take over.',
            heroStats: [
                { value: '\u2212' + decline + '%', label: 'Median wholesale price decline, 2030 (50%) \u2192 2045 (95%)', color: 'green' },
                { value: '$' + spread75 + '/MWh', label: 'Peak uncertainty band (P10\u2013P90) at 75\u201380% clean', color: 'amber' },
                p.heroStat3
            ],
            nuclearCalloutH4: 'Existing Nuclear at Risk: ' + p.nucFleet + ' fleet faces wholesale price decline',
            nuclearCalloutP: name + '\u2019s ' + p.nucFleet + ' nuclear fleet needs <strong>$25\u201330/MWh</strong> to cover operating costs. Our LMP model shows median wholesale prices fall through this floor as clean energy penetration increases. <a href="#nuclear-deepdive" style="color:#38bdf8;text-decoration:underline;">Jump to full analysis \u2193</a>',
            sect1HTML:
                nc('Price Drivers', 'rgba(14,165,233,0.08)', '#38bdf8',
                    'Fuel prices explain 88\u2013100% of wholesale cost variation \u2014 until they don\u2019t',
                    '<p>From 50% to 95% clean energy, the single biggest determinant of wholesale prices is the fossil fuel price assumption. At 50% clean, fuel level explains <strong>100%</strong> of the variance in average LMP.</p>' +
                    '<p>The gap between Low and High fuel scenarios in ' + name + ': at 80% clean, Low fuel yields $' + lowAt80 + '/MWh while High fuel yields $' + highAt80 + '/MWh \u2014 a ' + ratio80 + '\u00d7 difference.</p>',
                    'stat-badge-blue', 'Low fuel: $' + lowAt80 + '/MWh at 80% \u00b7 High fuel: $' + highAt80 + '/MWh at 80%') +
                nc(null, null, null,
                    'At \u226599.99% clean, the game changes completely',
                    '<p>When clean energy reaches \u226599.99%, fuel prices become irrelevant. Variance fragments across renewable costs, transmission, 45Q policy, and CO\u2082 price.</p><p>' + p.sect1Extra + '</p>',
                    'stat-badge-amber', 'At \u226599.99%: No single factor explains >15% of price variance'),
            sect2HTML:
                nc('Market Structure', 'rgba(34,197,94,0.08)', '#4ade80',
                    name + ' zero-price hours surge from ' + z0.toLocaleString() + ' to ' + zEnd.toLocaleString(),
                    '<p>At 50% clean, ' + name + ' sees about ' + z0.toLocaleString() + ' zero-or-negative-price hours per year. ' + p.zeroContext +
                    'By 95%, that figure hits <strong>' + z95.toLocaleString() + ' hours</strong>. At \u226599.99%, it\u2019s <strong>' + zEnd.toLocaleString() + ' hours</strong>.</p>' +
                    '<p>The curve accelerates sharply past 75%, as each incremental percentage of clean energy adds more zero-marginal-cost generation.</p>',
                    'stat-badge-green', z0.toLocaleString() + ' \u2192 ' + zEnd.toLocaleString() + ' zero-price hours: a ' + zMult + '\u00d7 increase') +
                nc(null, null, null,
                    'Scarcity events decline but never disappear',
                    '<p>Hours above $200/MWh drop from a median of ' + sc0 + ' at 50% clean to ' + scEnd + ' at \u226599.99%. Clean firm generation and storage smooth the worst supply crunches, but extreme weather still produces scarcity pricing.</p>',
                    'stat-badge-red', 'Scarcity hours: ' + sc0 + ' \u2192 ' + scEnd + ' median'),
            sect3HTML:
                nc('Price Structure', 'rgba(245,158,11,0.08)', '#fbbf24',
                    (iso === 'SPP' ? 'The inverted spread \u2014 off-peak exceeds peak' : 'The peak premium persists \u2014 and that\u2019s the arbitrage signal'),
                    '<p>' + p.peakInsight + '</p>' +
                    '<p>At 95% clean, peak is $' + pk95 + '/MWh and off-peak is $' + op95 + '/MWh. For storage operators and demand-response providers, the economic case for time-shifting energy remains strong through most of the transition.</p>',
                    'stat-badge-amber', 'Peak $' + pk0 + ' \u2192 $' + pk95 + ' \u00b7 Off-peak $' + op0 + ' \u2192 $' + op95 + ' at 95%') +
                nc(null, null, null,
                    (iso === 'SPP' ? 'Wind overgeneration defines the price structure' : 'Off-peak prices go negative first'),
                    '<p>At \u226599.99% clean, ' + (iso === 'SPP' ? 'peak' : 'off-peak') + ' median is <strong>$' + (iso === 'SPP' ? pkE : opE) + '/MWh</strong> while ' + (iso === 'SPP' ? 'off-peak' : 'peak') + ' is <strong>$' + (iso === 'SPP' ? opE : pkE) + '/MWh</strong>. Even in a fully decarbonized grid, timing mismatch creates a spread.</p>',
                    'stat-badge-blue', 'At \u226599.99%: peak $' + pkE + ', off-peak $' + opE),
            nuclearHeaderTag: p.nucTag,
            nuclearHeaderTitle: p.nucTitle,
            nuclearHeaderBody: p.nucIntro,
            nuclearChartTitle1: name + ' Wholesale Price vs. Nuclear Operating Cost \u2014 By Clean Energy %',
            nuclearChartTitle2: name + ' Hours Per Year at Zero or Negative Price \u2014 Nuclear Revenue Erosion',
            nuclearNarrativeHTML:
                nc('Viable Operation Threshold', 'rgba(239,68,68,0.06)', '#f87171', p.nucCard1Title, p.nucCard1, 'stat-badge-red', '') +
                nc('Historic Precedent', 'rgba(239,68,68,0.06)', '#f87171', p.nucCard2Title, p.nucCard2, 'stat-badge-amber', '') +
                nc(null, null, null, p.nucCard3Title, p.nucCard3, 'stat-badge-blue', '') +
                '<div class="nuclear-metric-row">' + p.nucMetrics.map(function(m) {
                    return '<div class="nuclear-metric"><div class="nuclear-metric-value" style="color:' + m.color + ';">' + m.value + '</div><div class="nuclear-metric-label">' + m.l1 + '<br>' + m.l2 + '</div></div>';
                }).join('') + '</div>' +
                '<div class="nuclear-paradox"><h3>' + p.nucParadoxTitle + '</h3>' + p.nucParadox + '</div>',
            synthesisHTML:
                '<div class="synthesis-card"><h3>The uncertainty paradox: widest where it matters most</h3>' +
                '<p>The P10\u2013P90 spread in ' + name + ' wholesale prices peaks at 75\u201380% clean energy \u2014 exactly the threshold range where most buyers are planning. At 80%, the gap is <strong>$' + Math.round(D.envelope.p90[6] - D.envelope.p10[6]) + '/MWh</strong>.</p>' +
                '<span class="stat-badge stat-badge-amber">$' + Math.round(D.envelope.p90[6] - D.envelope.p10[6]) + '/MWh spread at 80%</span></div>' +
                '<div class="synthesis-card"><h3>Three eras of the ' + name + ' wholesale market</h3>' +
                '<p><strong>Era 1: 50\u201375% (2030\u2013~2034)</strong> \u2014 Fossil sets the price. Markets function as today.</p>' +
                '<p><strong>Era 2: 75\u201395% (~2034\u20132045)</strong> \u2014 The transition zone. Uncertainty peaks. Zero-price hours become structural.</p>' +
                '<p><strong>Era 3: 95\u2013\u226599.99% (2045\u20132050)</strong> \u2014 The structural break. Revenue for all generators collapses.</p>' +
                '<span class="stat-badge stat-badge-green">2030\u20132034: business as usual \u00b7 2034\u20132045: transition \u00b7 2045+: new paradigm</span></div>',
            footerNote: 'All analysis on this page is ' + name + ' \u00b7 ' + p.scenarioCount + ' simulated scenarios \u00b7 Synthetic LMP model calibrated to ' + p.source
        };
    });
})();
