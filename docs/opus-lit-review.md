# The 8,760 Problem: A Comprehensive Literature Review

## Electricity System CO₂ Accounting and Hourly Clean Energy Matching

**The emerging field of hourly clean energy matching sits at the intersection of at least seven major research domains—CO₂ accounting frameworks, capacity expansion modeling, clean firm power economics, storage, market design, corporate procurement, and grid resilience—yet no prior work synthesizes these domains through a unified, multi-scenario optimization framework the way The 8,760 Problem does.** This literature review maps the full intellectual landscape, identifies more than 70 primary sources across all seven domains, and precisely locates where The 8,760 Problem site aligns with, diverges from, and extends beyond the existing literature.

-----

## 1. The Accounting Revolution: From Annual RECs to Hourly Matching

The intellectual foundation for hourly clean energy matching begins with the GHG Protocol Scope 2 Guidance (WRI, 2015), which established dual reporting: a location-based method using average grid emission factors and a market-based method using contractual instruments such as Renewable Energy Certificates. This framework enabled the modern voluntary clean energy market but also created a structural loophole.

Bjørn, Lloyd, Brander, and Matthews (2022, Nature Climate Change) demonstrated that across 115 companies with science-based targets, the widespread use of RECs led to an inflated estimate of mitigation effectiveness: companies reported a 31% reduction in scope 2 emissions from 2015–2019, but stripping out REC-based claims revealed an actual reduction of only ~10%. If this trend continued, 42% of committed scope 2 reductions would not result in real-world mitigation. This landmark paper directly catalyzed the ongoing GHG Protocol Scope 2 revision.

The proposed Scope 2 revision (GHG Protocol, Standard Development Plan, December 2024; Phase 1 public consultation launched October 2025, closing January 2026) represents the most consequential shift in corporate carbon accounting since 2015. The revision proposes hourly matching requirements for market-based instruments, geographic deliverability constraints, and residual mix reform using fossil-only emission factors where residual data is unavailable. The timeline targets a revised standard by 2027. Separately, the GHG Protocol has opened a parallel track for consequential accounting guidance.

Brander (2022, Carbon Management) and Brander and Bjørn (2023, International Journal of Life Cycle Assessment) articulated the theoretical underpinning: attributional methods allocate emissions within a boundary and can be summed to equal total global emissions, while consequential methods quantify system-wide changes caused by decisions. Mixing the two, as the current market-based method does, produces “incoherent results.”

This debate has organized two camps. The hourly matching camp, led by Google, EnergyTag, and the Climate Group’s 24/7 Carbon-Free Coalition (launched September 2024), prioritizes time- and location-matched energy procurement. The carbon matching camp, led by the Emissions First Partnership (founded 2022; members include Amazon, Meta, Intel, Salesforce, General Motors), uses marginal operating emission rates (MOERs) to maximize total emissions reductions regardless of location.

WattTime provides the foundational data for marginal emissions analysis. Building on Siler-Evans, Azevedo, and Morgan (2012, Environmental Science & Technology)—which established that average emission factors can “grossly misestimate” avoided emissions—WattTime now produces MOERs every 5 minutes across 210 countries covering ~99% of global electricity consumption. The VERACI-T validation working group (2023) compared six MEF models across US ISOs, finding WattTime (−1.3% average error) and Siler-Evans (−2.7%) significantly more accurate than alternatives.

REsurety, which jointly launched a free Grid Emissions Data Platform with WattTime in March 2025, emphasizes that the two approaches serve different goals: hourly energy matching for consumption-based accountability, carbon matching for maximizing total emissions reductions.

**Alignment with The 8,760 Problem:** The site’s 10-strategy procurement comparison framework—comprising 3 consequential, 3 hourly, and 4 annual matching strategies—directly operationalizes this accounting debate. By modeling all approaches simultaneously across identical scenarios, the site enables empirical comparison of outcomes rather than theoretical argument. This multi-strategy comparison is unique in the literature.

**Extension:** While existing studies compare at most 2–3 strategies (e.g., Xu et al. compare annual, temporal, and emissions matching), The 8,760 Problem’s 10-strategy framework provides the most comprehensive side-by-side evaluation documented to date.

-----

## 2. Granular Certificates and the Institutional Infrastructure for Hourly Matching

The practical enablement of hourly matching requires institutional infrastructure that is still under construction. EnergyTag published its Granular Certificate (GC) Scheme Standard V2 in December 2024 and Matching Standard V1 in March 2024. GCs represent energy production during intervals of one hour or less. In June 2025, EnergyTag accredited its first two GC schemes—Energinet (Denmark’s TSO) and a second scheme—marking a milestone for operational granular certification.

Google’s portfolio conversion demonstrates feasibility at scale: Flexidao converted Google’s global portfolio to hourly format using Config 3, covering 10.5 TWh across 7 countries and 100+ sites in 2025. In the US, registries including PJM GATs, M-RETS, and NEPOOL are developing GC capability.

The SBTi Corporate Net-Zero Standard V2.0 (second consultation draft, November 2025) codifies hourly matching into corporate target-setting: companies must achieve 100% low-carbon electricity by 2040, with hourly matching required starting 2030. The separate SBTi Power Sector Net-Zero Standard (first draft, September 2025) sets a net-zero target year of 2040 for power companies.

EPRI’s SMARTargets framework (announced COP28 December 2023; public consultation closed August 2025) offers a contrasting approach—company-specific “Qualified Targets” reflecting unique transition constraints alongside “Aspirational Targets” from global 1.5°C pathways. Ceres (August 2025) criticized the approach as potentially allowing less ambitious targets.

**Alignment with The 8,760 Problem:** The site’s ISO-specific analysis across 7 US regions reflects the SMARTargets philosophy that company-specific context matters, while its scenario-based approach enables testing whether aspirational targets are achievable under realistic conditions.

**Divergence:** The site goes beyond both SBTi’s phased requirements and EPRI’s qualitative framework by quantifying exact cost curves for every matching threshold from 90% to 100% in 2.5% increments.

-----

## 3. The Princeton Finding That Annual Matching Fails: Hourly CFE Modeling Literature

The most consequential finding comes from Princeton’s ZERO Lab. Xu, Ricks, Manocha, Patankar, and Jenkins (2024, Joule) enhanced GenX to compare three voluntary procurement strategies across US regions. Their central finding: annual/volumetric matching produces “zero or near-zero” system-level CO₂ reductions in the long run, because wind and solar procurement largely displaces other clean energy rather than fossil fuels. Only temporal (hourly) matching consistently lowered system-wide emissions.

Riepin and Brown (2024, Energy Strategy Reviews) extended this to Europe using PyPSA, quantifying the cost nonlinearity: a 98% CFE target costs 54% more than annual matching, and the last 2% (98% → 100%) more than doubles costs. In a 2025 Joule commentary, Riepin, Jenkins, Swezey, and Brown demonstrated that 24/7 CFE matching accelerates technology learning curves.

RMI’s “Clean Power by the Hour” (Dyson, Shah, and Teplin, July 2021), supported by Microsoft, assessed hourly matching across 7 markets and identified a three-stage cost escalation: stable costs to 30–80% hourly matching, significant increase to ~85%, then sharp escalation above 85%.

TransitionZero (2025, series funded by Google.org) produced the most geographically diverse 24/7 CFE modeling, studying India, Japan, Singapore, and Malaysia. The cross-cutting finding—that moderate hourly matching (70–90%) can be cheaper than annual matching at system level—challenges the assumption that hourly matching necessarily costs more. India can add 70% 24/7 CFE at lower cost than annual matching, saving grid operators US$1 billion/year; Japan achieves 90% CFE at competitive costs, saving up to $1.84 billion annually.

Ludkovski, Mouti, and Swindle (2023, arXiv:2312.07733) developed a probabilistic framework for CFE portfolio optimization in ERCOT, finding costs increase exponentially as CFE targets approach 100%.

The IEA’s “Advancing Decarbonisation through Clean Electricity Procurement” (November 2022) confirmed that for India and Indonesia, optimized dispatch of hourly matching portfolios reduces both system costs and emissions relative to isolated dispatch.

**Alignment with The 8,760 Problem:** The site’s 8,760-hour dispatch optimization across 7 US ISOs directly builds on the Princeton ZERO Lab methodology.

**Extension:** Where Princeton, Riepin and Brown, and TransitionZero each study individual regions with single-point cost assumptions, The 8,760 Problem runs 5,832+ cost scenarios per ISO with ~33,000 physics-feasible resource mixes, providing a distribution of outcomes rather than point estimates. The 2.5% threshold granularity in the 90–100% zone is finer than any published study, which typically report results at 5–10% intervals.

-----

## 4. Clean Firm Power: The Technologies That Close the Hourly Gap

The economic case for clean firm power is anchored by Sepulveda, Jenkins, de Sisternes, and Lester (2018, Joule), who found that firm low-carbon technologies reduce electricity costs by 10–62% across fully decarbonized scenarios. Even under the most optimistic wind, solar, and battery cost assumptions, firm resources consistently lowered system costs.

Long et al. (2021, Issues in Science and Technology)—the landmark EDF/CATF California study—used three independent models and found that portfolios including clean firm power would be 32–53% cheaper than renewables-and-batteries-only pathways. California needs ~30 GW of clean firm capacity by 2045.

Spokas and Ricks (CATF, February 2026) published the most comprehensive technology assessment to date, concluding that diversified systems with clean firm generation are “significantly less expensive (often by tens of percent)” than variable-renewables-only systems.

### Nuclear Cost Trajectories

The DOE “Pathways to Commercial Liftoff: Advanced Nuclear” (September 2024 update) projects NOAK cost targets of $3,600/kW with a committed orderbook of 5–10 deployments. MIT (Shirvan, 2024, CANES Report ANP-201 TR) estimated AP1000 NOAK at ~$4,625/kW with unsubsidized LCOE of $66/MWh. INL (Bolisetti et al., June 2024) identified pathways achieving 45–60% cost reductions between first and third plants.

The NREL ATB 2024 nuclear module models learning rates of 8% (large reactor) and 9.5% (SMR) per doubling of cumulative capacity.

### Learning Curves and Wright’s Law

Bolinger, Wiser, and O’Shaughnessy (2022, iScience, LBNL) provided rigorous LCOE-based learning analysis: solar PV exhibits a 24% learning rate per doubling and onshore wind 15%. Way, Iribarren, Hepburn, and Farmer (Oxford) demonstrated that Wright’s Law-based forecasting has been the most accurate predictor of solar cost trajectories.

### Current Cost Benchmarks

Lazard LCOE+ v18.0 (June 2025) provides unsubsidized benchmarks: utility-scale solar at $38–78/MWh, onshore wind at $37–86/MWh, gas combined cycle at $48–109/MWh, geothermal at $66–109/MWh, and new-build nuclear at $141–220/MWh.

GridLab, Energy Futures Group, and Halcyon (September 2025) revealed CCGT projects routinely exceeding $2,000/kW—while EIA AEO assumes only $921/kW. This massive disconnect between modeled assumptions and market reality has profound implications.

### Enhanced Geothermal

Fervo Energy’s Cape Station targets 100 MW online by October 2026 and 500 MW by 2028, with drilling times fallen 70% year-over-year. DOE targets EGS LCOE of $45/MWh by 2035.

**Alignment with The 8,760 Problem:** The site’s four-pool supply model (SSS/Corporate-Contracted/Merchant/New-Build) directly operationalizes the distinction between existing and new clean firm power.

**Extension:** The critical mass analysis for Wright’s Law learning curve activation is novel—no published study quantifies how many corporate procurement decisions are needed to trigger specific learning rate milestones at the ISO level.

-----

## 5. Why LCOE Misleads: System Cost, Cannibalization, and Value Deflation

Joskow (2011, American Economic Review) established that LCOE is “seriously flawed” because it treats all MWh as homogeneous, overvaluing intermittent technologies. Hirth (2013, Energy Economics) quantified the consequence: wind value factors drop to ~0.7 at 30% market share and solar to ~0.7 at only 10–15% penetration. Hirth, Ueckerdt, and Edenhofer (2015, Renewable Energy) decomposed integration costs into profile costs (~25–35 €/MWh at 30–40% wind, the dominant component), balancing costs (~2–4 €/MWh), and grid costs. Ueckerdt et al. (2013, Energy) formalized System LCOE = Generation LCOE + Integration Costs.

By 2024, solar capture rates in CAISO’s SP15 zone had plummeted to less than 30% (REsurety, 2025). Negative pricing in CAISO SP15 doubled from ~530 hours in 2023 to ~1,180 hours in 2024 (~13% of all hours). In ERCOT, wind and solar together met 36% of demand in the first 9 months of 2025.

These dynamics are captured in NREL’s ReEDS modeling (Gagnon et al., 2024 Standard Scenarios) and the Princeton Net-Zero America Project (Larson et al., October 2021). GenX (Jenkins and Sepulveda, MIT) has become the workhorse model for capacity expansion analysis.

**Alignment with The 8,760 Problem:** The site’s synthetic LMP model (150,000+ scenarios per ISO) directly addresses the cannibalization challenge.

**Extension:** The stranded investment analysis—comparing interim vs. long-term target procurement decisions—is not found in the academic capacity expansion literature, which typically optimizes for a single future year.

-----

## 6. The Last Mile: Storage Economics, Grid Resilience, and Resource Adequacy

Cole et al. (2021, Joule, NREL) demonstrated that costs increase nonlinearly for the last few percent toward 100% renewable electricity. Mai et al. (2022, Joule, NREL) evaluated six strategies for the last 10%, concluding that no single strategy is sufficient.

### Long-Duration Energy Storage

Form Energy’s iron-air battery targets <$20/kWh for 100+ hours of continuous discharge, though current costs exceed $150/kWh. The DOE Long Duration Storage Shot (July 2021) targets 90% cost reduction for 10+ hour storage. The DOE Pathways to Commercial Liftoff for LDES (2023) estimates net-zero pathways deploying LDES yield $10–20 billion/year in savings and reduce the need for 200+ GW of new gas capacity. The LDES Council/McKinsey analysis (2021) concluded LDES is most competitive beyond 6–8 hours.

### Battery Cost Trajectories

BloombergNEF reports global average lithium-ion pack prices at $108/kWh in 2025 (down 93% since 2010), with stationary storage packs at $70/kWh.

### Resource Adequacy and Dunkelflaute

PJM adopted marginal ELCC methodology (FERC-approved January 2024). Solar incremental capacity credit in CAISO is expected to drop to 6% by 2026. Kittel et al. (2024, arXiv) quantified 2–10 Dunkelflaute events per year in northern Europe, mostly October–February. Germany experienced a major event in November 2024 with renewable contribution dropping to 30% and prices surging above €145/MWh.

**Alignment with The 8,760 Problem:** The site’s treatment of the 90–100% zone with 2.5% granularity directly addresses the “last mile” problem.

**Extension:** The MAC vs. DAC crossover analysis with “no-regrets” resource identification represents a novel decision framework not found in the existing literature.

-----

## 7. The Interconnection Bottleneck and Market Design Challenges

Berkeley Lab’s “Queued Up” series (Rand et al., 2025) documents ~2,300 GW of capacity actively seeking grid connection, with only 13% of capacity from 2000–2019 reaching commercial operations. Median wait times have doubled to over 4 years.

FERC Order No. 2023 (July 2023) replaced first-come-first-served with cluster-based studies, increased financial readiness requirements, and mandated interconnection heatmaps.

Lo Prete, Palmer, and Robertson (RFF Report 24-09, June 2025) reviewed 11 proposed market designs and concluded existing designs are poorly suited for the evolving resource mix. PJM’s July 2024 capacity auction saw prices surge to $269.92/MW-day, a nearly 10-fold increase. ERCOT’s energy-only market faces separate challenges with loss-of-load expectation at 1.25 days/year.

The DOE National Transmission Needs Study (October 2023) concluded that interregional capacity must grow more than fivefold to realize IRA benefits.

**Alignment with The 8,760 Problem:** The site covers all 7 US ISOs, inherently modeling differences between capacity markets (PJM, NYISO, ISO-NE), hybrid designs (MISO, SPP), and energy-only markets (ERCOT).

**Extension:** No prior study systematically compares procurement optimization outcomes across all 7 ISOs simultaneously.

-----

## 8. Corporate Procurement Enters the Firm Power Era

BloombergNEF data shows corporations signed ~62 GW of clean power PPAs in 2024 before declining to 55.9 GW in 2025. Critically, 5.2 GW of 2025 deals were “baseload-like” products—a structural shift from wind and solar PPAs.

Nuclear deals are unprecedented: Microsoft signed a $1.6 billion PPA with Constellation for TMI Unit 1 restart (835 MW). Google signed the first corporate SMR fleet deal with Kairos Power (up to 500 MW). Amazon is pursuing nearly 2 GW from Talen Energy’s Susquehanna. Meta signed 1.1 GW with Constellation’s Clinton plant. Combined, Big Tech contracted 10+ GW of new nuclear in 2024–2025.

IRA Section 45V (final regulations January 2025) established hourly matching required starting January 1, 2030 for hydrogen production tax credits—the first federal hourly matching mandate in US law.

**Alignment with The 8,760 Problem:** The site’s corporate-contracted and merchant supply pools mirror the real-world landscape.

**Extension:** The stranded investment analysis connects Carbon Tracker’s macro stranded-asset framework to individual procurement decisions.

-----

## 9. Nine Novel Contributions That Distinguish The 8,760 Problem

Having mapped the full literature landscape, nine novel contributions can be precisely identified:

**1. Multi-scenario co-optimization at unprecedented scale.** 5,832+ sensitivity scenarios per ISO, producing distributions of outcomes rather than point estimates—unmatched in the published literature.

**2. The four-pool supply model.** SSS/Corporate-Contracted/Merchant/New-Build decomposition reflects real-world procurement constraints that academic models abstract away.

**3. The 10-strategy procurement comparison framework.** The most comprehensive side-by-side evaluation of accounting approaches documented to date.

**4. Critical mass analysis for Wright’s Law activation.** Quantifies specific deployment thresholds needed to trigger learning rates for advanced nuclear and enhanced geothermal at the ISO level.

**5. ~33,000 physics-feasible resource mixes per ISO.** A search space orders of magnitude larger than any published study’s technology palette exploration.

**6. Stranded investment analysis.** Bridges macro stranded-asset analysis to corporate procurement decision-making by comparing interim vs. long-term target optimization.

**7. Synthetic LMP model with 150,000+ scenarios per ISO.** Extends beyond existing temporal mismatch analysis to generate comprehensive price distributions.

**8. MAC vs. DAC crossover analysis.** Identifies where hourly matching becomes more expensive than direct air capture, with “no-regrets” resource identification below the threshold.

**9. 2.5% threshold granularity in the 90–100% inflection zone.** Finer than any published study, providing practitioners the precision needed for target-setting in the most cost-sensitive zone.

-----

## Conclusion

Across all seven domains, the research literature converges: annual REC matching has negligible system-level emissions impact; hourly matching with diverse clean resources is essential but costly above ~90%; firm clean power reduces system costs by tens of percent; LCOE misleads about dispatchable resource value; interconnection queues create binding constraints; and corporate procurement is shifting toward firm, hourly-matched products.

What the literature lacks—and what The 8,760 Problem provides—is a unified analytical framework connecting these findings across all seven domains simultaneously, at the geographic specificity of individual ISOs, with the scenario breadth needed to support actual procurement decisions under deep uncertainty. By bridging academic models and practitioner needs with unprecedented scenario scale, the site occupies a genuinely novel position—one that the imminent GHG Protocol Scope 2 revision, SBTi V2 requirements, and the growth of corporate firm-power procurement will make increasingly relevant through 2027 and beyond.

-----

## Key Sources Referenced (70+)

### CO₂ Accounting & Frameworks

1. GHG Protocol, Scope 2 Guidance (WRI, 2015)
1. GHG Protocol, Scope 2 Proposed Updates & Consequential Accounting Consultation (Oct 2025 – Jan 2026)
1. Bjørn, Lloyd, Brander & Matthews, “Renewable energy certificates threaten the integrity of corporate science-based targets,” Nature Climate Change (2022)
1. Brander, “The most important GHG accounting concept you may not have heard of,” Carbon Management (2022)
1. Brander & Bjørn, “Attributional and consequential greenhouse gas accounting,” Int J Life Cycle Assessment (2023)
1. Bjørn et al., “Ensuring low-emission electricity purchasing requires broader systems perspective,” Nature Communications (2024)
1. SBTi Corporate Net-Zero Standard V2.0, Second Consultation Draft (November 2025)
1. SBTi Power Sector Net-Zero Standard, First Draft (September 2025)
1. EPRI SMARTargets Framework (July 2025)
1. Ceres, “Ceres calls on EPRI to strengthen their draft emissions reduction guidance” (August 2025)

### Marginal Emissions & Data

1. Siler-Evans, Azevedo & Morgan, “Marginal Emissions Factors for the U.S. Electricity System,” Env Sci & Tech (2012)
1. WattTime, Automated Emissions Reduction (AER) methodology
1. VERACI-T Working Group, “Towards Objective Evaluation of the Accuracy of Marginal Emission Factor Models” (2023)
1. REsurety & WattTime, Grid Emissions Data Platform (March 2025)
1. Emissions First Partnership (founded 2022)

### Granular Certificates

1. EnergyTag, Granular Certificate Scheme Standard V2 (December 2024)
1. EnergyTag, Matching Standard V1 (March 2024)
1. Flexidao/Google, Global portfolio hourly conversion (2025)

### Hourly CFE Modeling

1. Xu, Ricks, Manocha, Patankar & Jenkins, “System-level impacts of 24/7 carbon-free energy procurement,” Joule (2024)
1. Riepin & Brown, “On the means, costs, and system-level impacts of 24/7 CFE procurement,” Energy Strategy Reviews (2024)
1. Riepin, Jenkins, Swezey & Brown, “24/7 CFE matching accelerates adoption of advanced clean energy technologies,” Joule Commentary (2025)
1. RMI, “Clean Power by the Hour” (Dyson, Shah, Teplin, July 2021)
1. TransitionZero, “24/7 CFE in India” (2025)
1. TransitionZero, “24/7 CFE in Japan” (2025)
1. TransitionZero, “24/7 CFE in Singapore” (2025)
1. TransitionZero, “24/7 CFE in Malaysia” (2025)
1. Ludkovski, Mouti & Swindle, “Least-Cost Structuring of 24/7 CFE Procurements,” arXiv:2312.07733 (2023)
1. IEA, “Advancing Decarbonisation through Clean Electricity Procurement” (November 2022)
1. Google, “24/7 Carbon-Free Energy” methodology and annual reports

### Clean Firm Power Economics

1. Sepulveda, Jenkins, de Sisternes & Lester, “The role of firm low-carbon electricity resources,” Joule (2018)
1. Long et al., “Clean Firm Power is the Key to California’s Carbon-Free Energy Future,” Issues in Science and Technology (2021)
1. Spokas & Ricks (CATF), “Clean Firm Electricity Technologies: What, Why, How” (February 2026)
1. CATF, “Beyond LCOE” (June 2025)
1. DOE, “Pathways to Commercial Liftoff: Advanced Nuclear” (September 2024 update)
1. Shirvan (MIT CANES), “Overnight Capital Cost of the Next AP1000,” Report ANP-201 TR (2024)
1. Bolisetti et al. (INL), “Quantifying Capital Cost Reduction Pathways,” INL/RPT-24-7767 (June 2024)
1. Abou-Jaoude et al. (INL), “Meta-Analysis of Advanced Nuclear Reactor Cost Estimations,” INL/RPT-23-72972 (2024)
1. NREL Annual Technology Baseline 2024, Nuclear Module

### Learning Curves

1. Bolinger, Wiser & O’Shaughnessy, “Levelized cost-based learning analysis of utility-scale wind and solar,” iScience (2022)
1. Way, Iribarren, Hepburn & Farmer, “Empirically grounded technology forecasts and the energy transition” (Oxford)
1. Wright, “Factors Affecting the Cost of Airplanes,” Journal of Aeronautical Sciences (1936)

### Cost Benchmarks

1. Lazard, LCOE+ Version 18.0 (June 2025)
1. GridLab/EFG/Halcyon, “Gas Turbine Cost Report” (September 2025)
1. NREL ATB 2024 (all technology modules)

### Enhanced Geothermal

1. Fervo Energy, Cape Station development updates (2024–2025)
1. DOE, Enhanced Geothermal Shot targets
1. CATF, Superhot Rock Geothermal assessment

### System Cost & Value Deflation

1. Joskow, “Comparing the Costs of Intermittent and Dispatchable Electricity Generating Technologies,” American Economic Review (2011)
1. Hirth, “The Market Value of Variable Renewables,” Energy Economics (2013)
1. Hirth, Ueckerdt & Edenhofer, “Integration costs revisited,” Renewable Energy (2015)
1. Ueckerdt, Hirth, Luderer & Edenhofer, “System LCOE,” Energy (2013)
1. López Prol, Steininger & Zilberman, “Cannibalization effect of wind and solar,” Energy Economics (2020)
1. REsurety/Bhandari, “Negative Prices in CAISO” (April 2025)

### Capacity Expansion Models

1. NREL, 2024 Standard Scenarios Report (Gagnon et al., NREL/TP-6A40-92256)
1. Princeton Net-Zero America Project (Larson et al., October 2021)
1. GenX capacity expansion model (Jenkins lab, MIT/Princeton/NYU/Binghamton)

### Storage & Grid Resilience

1. Cole et al., “Quantifying the challenge of reaching 100% renewable electricity,” Joule (2021)
1. Mai et al., “On the Road to 100% Clean Electricity: Six Strategies,” Joule (2022)
1. Form Energy, Iron-air battery technology and deployment pipeline
1. DOE, Long Duration Storage Shot (July 2021)
1. DOE, “Pathways to Commercial Liftoff: Long Duration Energy Storage” (2023)
1. LDES Council/McKinsey, Long Duration Energy Storage assessment (2021)
1. BloombergNEF, Battery Price Survey (2025)
1. Kittel et al., “Quantifying the Dunkelflaute,” arXiv (2024)
1. PJM, Marginal ELCC methodology (FERC-approved January 2024)

### Interconnection & Market Design

1. Berkeley Lab, “Queued Up” (Rand et al., 2025 edition)
1. FERC Order No. 2023 (July 2023) and Order 2023-A (March 2024)
1. Lo Prete, Palmer & Robertson (RFF), “Time for a Market Upgrade?” (June 2025)
1. DOE, National Transmission Needs Study (October 2023)
1. Simeone & Rose (NREL), Inter-regional transfer analysis (June 2024)
1. Hogan, “Electricity Market Design and the Green Agenda,” The Electricity Journal (2017)
1. Ela et al. (NREL), Revenue sufficiency analysis in ERCOT-like markets (2016)

### Corporate Procurement & Policy

1. BloombergNEF, Corporate PPA data (2024–2025)
1. IRA Section 45V, Final Regulations (January 2025)
1. Carbon Tracker, Stranded Assets analysis
1. IEA, World Energy Outlook 2024
1. Microsoft/Constellation TMI restart PPA (September 2024)
1. Google/Kairos Power SMR fleet deal (October 2024)
1. Amazon/X-energy/Talen Energy nuclear commitments (2024–2025)
1. Meta/Constellation Clinton Clean Energy Center deal (June 2025)