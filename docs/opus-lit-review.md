# The 8,760 Problem: Literature Review & References

**A comprehensive review of electricity system CO₂ accounting and hourly clean energy matching**

Academic foundations and cited works underpinning the 8,760 Problem analysis. This review maps the full intellectual landscape across seven research domains — CO₂ accounting frameworks, granular certificates, hourly CFE modeling, clean firm power economics, system cost analysis, storage and grid resilience, interconnection and market design, and corporate procurement — and identifies where The 8,760 Problem aligns with, diverges from, and extends existing literature.

-----

## 1. The Accounting Revolution: From Annual RECs to Hourly Matching

The intellectual foundation for hourly clean energy matching begins with the GHG Protocol Scope 2 Guidance (WRI, 2015) [[1]](#ref-1), which established dual reporting: a **location-based method** using average grid emission factors and a **market-based method** using contractual instruments such as Renewable Energy Certificates. This framework enabled the modern voluntary clean energy market but also created a structural loophole.

The inadequacy of annual renewable energy certificates for verifying actual grid decarbonization has been widely documented. Xu et al. (2024) demonstrate that annual matching can overstate clean energy coverage by 15–40% in regions with high solar penetration, because midday surplus certificates mask nighttime fossil consumption [[2]](#ref-2). Bjørn, Lloyd, Brander, and Matthews (2022) showed the consequences at scale: across 115 companies with science-based targets, the widespread use of RECs led to an inflated estimate of mitigation effectiveness — companies reported a 31% reduction in scope 2 emissions from 2015–2019, but stripping out REC-based claims revealed an actual reduction of only ~10%. If this trend continued, 42% of committed scope 2 reductions would not result in real-world mitigation [[3]](#ref-3). This landmark paper directly catalyzed the ongoing GHG Protocol Scope 2 revision.

The proposed Scope 2 revision (GHG Protocol, Standard Development Plan, December 2024; Phase 1 public consultation launched October 2025, closing January 2026) represents the most consequential shift in corporate carbon accounting since 2015. The revision proposes hourly matching requirements for market-based instruments, geographic deliverability constraints, and residual mix reform using fossil-only emission factors where residual data is unavailable. The timeline targets a revised standard by 2027 [[4]](#ref-4).

Brander (2022) and Brander and Bjørn (2023) have articulated the theoretical underpinning: **attributional methods** allocate emissions within a boundary and can be summed to equal total global emissions, while **consequential methods** quantify system-wide changes caused by decisions. Mixing the two, as the current market-based method does, produces “incoherent results” [[5]](#ref-5) [[6]](#ref-6). Bjørn et al. (2024, *Nature Communications*) further demonstrated that buyer-level marginal claims can sum to 2–3× actual grid reductions, reinforcing the need for methodological consistency [[7]](#ref-7).

This debate has organized two camps. The **hourly matching camp**, led by Google, EnergyTag, and the Climate Group’s 24/7 Carbon-Free Coalition (launched September 2024), prioritizes time- and location-matched energy procurement. The **carbon matching camp**, led by the Emissions First Partnership (founded 2022; members include Amazon, Meta, Intel, Salesforce, General Motors), uses marginal operating emission rates (MOERs) to maximize total emissions reductions regardless of location.

Google’s 24/7 CFE program, the largest corporate hourly matching initiative, reported achieving 64% average hourly CFE across its global operations in 2022 (rising to 66% in 2024), with significant regional variation [[8]](#ref-8). Their published methodology uses hourly generation data matched against facility-level consumption and has become a de facto industry standard. The UN 24/7 CFE Compact, launched at COP26, provides a multilateral framework for scaling hourly matching globally [[9]](#ref-9).

WattTime provides the foundational data for marginal emissions analysis. Building on Siler-Evans, Azevedo, and Morgan (2012) — which established that average emission factors can grossly misestimate avoided emissions [[10]](#ref-10) — WattTime now produces MOERs every 5 minutes across 210 countries. The VERACI-T validation working group (2023) compared six MEF models across US ISOs, finding WattTime (−1.3% average error) and Siler-Evans (−2.7%) significantly more accurate than alternatives [[11]](#ref-11).

REsurety, which jointly launched a free Grid Emissions Data Platform with WattTime in March 2025, emphasizes that the two approaches serve different goals: hourly energy matching takes responsibility for electricity consumed, while carbon matching maximizes total emissions reductions as fast as possible.

*Alignment with The 8,760 Problem:* The site’s 10-strategy procurement comparison framework — comprising 3 consequential, 3 hourly, and 4 annual matching strategies — directly operationalizes this accounting debate. By modeling all approaches simultaneously across identical scenarios, the site enables empirical comparison of outcomes rather than theoretical argument. This multi-strategy comparison is unique in the literature.

-----

## 2. Granular Certificates and the Institutional Infrastructure for Hourly Matching

The practical enablement of hourly matching requires institutional infrastructure that is still under construction. **EnergyTag**, the independent nonprofit, published its Granular Certificate (GC) Scheme Standard V2 in December 2024 and Matching Standard V1 in March 2024. GCs represent energy production during intervals of one hour or less and can be implemented through three configurations: direct issuance by local EAC bodies (Config 1), independent GC issuance alongside existing EAC systems (Config 2), or conversion of monthly EACs using hourly meter data (Config 3, most widely available today). In June 2025, EnergyTag accredited its first two GC schemes — Energinet (Denmark’s TSO) and a second scheme — marking a milestone for operational granular certification [[12]](#ref-12).

Google’s portfolio conversion demonstrates feasibility at scale: Flexidao converted Google’s global portfolio to hourly format using Config 3, covering 10.5 TWh across 7 countries and 100+ sites in 2025. In the US, registries including PJM GATs, M-RETS, and NEPOOL are developing GC capability.

The **SBTi Corporate Net-Zero Standard V2.0** (second consultation draft, November 2025) codifies hourly matching into corporate target-setting: companies must achieve 100% low-carbon electricity by 2040, with hourly matching required starting 2030. Geographic matching within the same grid region is mandatory, and additionality requires facilities commissioned within 10 years (shifting to 5 years by 2035) [[13]](#ref-13). The separate SBTi Power Sector Net-Zero Standard (first draft, September 2025) sets a net-zero target year of 2040 for power companies [[14]](#ref-14).

EPRI’s **SMARTargets** framework (announced COP28 December 2023; public consultation closed August 2025) offers a contrasting approach — company-specific “Qualified Targets” reflecting unique transition constraints alongside “Aspirational Targets” from global 1.5°C pathways [[15]](#ref-15). Ceres (August 2025) criticized the approach as potentially allowing less ambitious targets [[16]](#ref-16).

*Alignment with The 8,760 Problem:* The site’s ISO-specific analysis across 7 US regions reflects the SMARTargets philosophy that company-specific context matters, while its scenario-based approach enables testing whether aspirational targets are achievable under realistic conditions. *Divergence:* The site goes beyond both SBTi’s phased requirements and EPRI’s qualitative framework by quantifying exact cost curves for every matching threshold from 90% to 100% in 2.5% increments.

-----

## 3. The Princeton Finding That Annual Matching Fails: Hourly CFE Modeling Literature

The most consequential finding in the hourly matching literature comes from Princeton’s ZERO Lab. Xu, Ricks, Manocha, Patankar, and Jenkins (2024, *Joule*) enhanced the GenX capacity expansion model to compare three voluntary procurement strategies — annual matching, temporal (hourly) matching, and emissions matching — across US regions. Their central finding: annual/volumetric matching produces “zero or near-zero” system-level CO₂ reductions in the long run, because wind and solar procurement largely displaces other clean energy rather than fossil fuels. Only temporal (hourly) matching consistently lowered system-wide emissions [[17]](#ref-17). In a separate analysis, Xu et al. (2024) used GenX to show that when both RPS compliance and voluntary C&I demand draw on the same regional supply curve, costs escalate non-linearly as combined demand approaches the buildable capacity frontier [[18]](#ref-18).

Riepin and Brown (2024, *Energy Strategy Reviews*) extended this to Europe using PyPSA, quantifying the cost nonlinearity: a 98% CFE target costs 54% more than annual matching, and the last 2% (98% → 100%) more than doubles costs [[19]](#ref-19). In a 2025 *Joule* commentary, Riepin, Jenkins, Swezey, and Brown demonstrated that 24/7 CFE matching accelerates technology learning curves [[20]](#ref-20).

RMI’s “Clean Power by the Hour” (Dyson, Shah, and Teplin, July 2021), supported by Microsoft, assessed hourly matching across 7 markets and identified a three-stage cost escalation: stable costs to 30–80% hourly matching, significant increase to ~85%, then sharp escalation above 85% [[21]](#ref-21). The IEA’s analysis of 24/7 CFE methodologies and case studies further validates this three-phase pattern [[22]](#ref-22).

TransitionZero (2025, series funded by Google.org) produced the most geographically diverse 24/7 CFE modeling, studying India, Japan, Singapore, and Malaysia. The cross-cutting finding — that moderate hourly matching (70–90%) can be cheaper than annual matching at system level — challenges the assumption that hourly matching necessarily costs more [[23]](#ref-23).

The “last few percent” phenomenon is well-documented: Denholm et al. (2021) show that costs increase non-linearly as clean electricity targets approach 100% [[24]](#ref-24). However, the 8,760 Problem optimizer results show a more nuanced picture. Average abatement costs remain well below the Rennert social cost of carbon ($185/ton) through 99% hourly matching across all seven ISOs. But the stepwise marginal analysis reveals critical decision boundaries: the cost escalation is non-monotonic and region-specific, driven by resource mix transitions at each threshold.

Gillenwater (2008) provides the foundational supply-demand framework for voluntary REC markets, showing that when the compliance market is “long” (supply exceeds mandate), voluntary retirement has near-zero effect on new investment. Only when combined demand creates genuine scarcity does voluntary procurement drive additional generation [[25]](#ref-25). O’Shaughnessy et al. (2025) document 319 million MWh in voluntary procurement (44% of non-hydro renewable sales), with long-term PPAs — not spot RECs — as the primary channel through which corporate demand signals new capacity [[26]](#ref-26).

*Alignment with The 8,760 Problem:* The site’s scarcity model operationalizes these insights: RPS-mandated build consumes the cheapest tiers of the regional supply stack first, with voluntary corporate demand riding on top. *Extension:* The 2.5% threshold granularity in the 90–100% inflection zone is finer than any published study, providing practitioners the precision needed for target-setting in the most cost-sensitive zone.

-----

## 4. Clean Firm Power: The Technologies That Close the Hourly Gap

The economic case for clean firm power is anchored by Sepulveda, Jenkins, de Sisternes, and Lester (2018, *Joule*), who evaluated nearly 1,000 cases using GenX and found that firm low-carbon technologies reduce electricity costs by 10–62% across fully decarbonized scenarios. Even under the most optimistic wind, solar, and battery cost assumptions, firm resources consistently lowered system costs. Without them, storage capacity requirements explode from 29–380 GWh to 320–1,160 GWh [[27]](#ref-27).

Long et al. (2021, *Issues in Science and Technology*) — the landmark EDF/CATF California study — used three independent models (GenX, Stanford, E3) with harmonized assumptions and found that portfolios including clean firm power would be 32–53% cheaper than renewables-and-batteries-only pathways. California needs ~30 GW of clean firm capacity by 2045; without it, the state requires ~160 GW of short-term battery storage and must triple transmission lines [[28]](#ref-28).

The Princeton Net-Zero America study (Larson et al., 2021) identifies firm, dispatchable clean generation as essential for achieving deep decarbonization cost-effectively [[29]](#ref-29). Jenkins et al. (2018) demonstrate that removing firm low-carbon resources from the generation portfolio increases electricity system costs by 10–62% under high decarbonization scenarios [[30]](#ref-30). Spokas and Ricks (CATF, February 2026) published the most comprehensive technology assessment to date in “Clean Firm Electricity Technologies: What, Why, How,” concluding that diversified systems with clean firm generation are “significantly less expensive (often by tens of percent)” than variable-renewables-only systems. The companion report “Beyond LCOE” (CATF, June 2025) argues that LCOE-focused decisions systematically underinvest in dispatchable clean firm power [[31]](#ref-31).

### Nuclear Cost Trajectories

Nuclear power, the largest existing source of firm clean generation, faces economic pressure from cheap natural gas and subsidized renewables. The EIA reports existing nuclear fleet operating costs of $28–35/MWh, making it cost-competitive with wholesale market prices in most regions [[32]](#ref-32). New nuclear construction, however, faces significantly higher costs.

The DOE “Pathways to Commercial Liftoff: Advanced Nuclear” (September 2024 update) projects that US nuclear capacity could triple from ~100 GW to ~300 GW by 2050, with NOAK cost targets of $3,600/kW. A committed orderbook of 5–10 deployments of at least one design is the essential first step [[33]](#ref-33). Shirvan at MIT (2024, CANES Report ANP-201 TR) independently estimated AP1000 NOAK overnight capital cost at ~$4,625/kW with an unsubsidized LCOE of $66/MWh, declining to ~$2,900–3,000/kW for the 10th unit [[34]](#ref-34). The INL “Quantifying Capital Cost Reduction Pathways” report (Bolisetti et al., June 2024) identified pathways achieving 45–60% cost reductions between first and third plants [[35]](#ref-35).

The NREL ATB 2024 nuclear module, informed by Abou-Jaoude et al.’s meta-analysis (2024), models cost evolution from three initial 2030 cost quartiles through learning rate assumptions. Learning rates of 8% (large reactor) and 9.5% (SMR) per doubling of cumulative capacity are used, though historical US/Western experience shows “negative learning” (cost increases), while sequential builds of identical designs (e.g., Korean APR1400 in UAE) achieved 40% labor cost reductions between Units 1 and 4 [[36]](#ref-36) [[37]](#ref-37).

The real-world evidence is sobering. NuScale’s UAMPS project saw costs escalate from $5.3 billion to $9.3 billion (75% increase) before cancellation [[38]](#ref-38). Vogtle Units 3&4 cost $30–36.8 billion versus the original $14 billion estimate [[39]](#ref-39). Frontiers in Energy Research (2022) estimates NGCC with post-combustion CCS at FOAK ~$103/MWh declining to NOAK ~$92/MWh [[40]](#ref-40).

Our cost model uses $90/MWh for new-build clean firm across all regions, consistent with NREL ATB 2024 estimates for advanced nuclear or enhanced geothermal systems [[41]](#ref-41).

### Learning Curves and Wright’s Law

The theoretical framework for technology cost projections rests on Wright’s Law (1936), which posits that unit costs decline by a fixed percentage with each doubling of cumulative production [[42]](#ref-42). Bolinger, Wiser, and O’Shaughnessy (2022, *iScience*, LBNL) provided the most rigorous LCOE-based learning analysis: solar PV exhibits a 24% learning rate per doubling and onshore wind 15%, with both experiencing an accelerated learning period of 40–45% through 2020 [[43]](#ref-43). Way, Iribarren, Hepburn, and Farmer (Oxford) demonstrated that Wright’s Law-based forecasting has been the most accurate predictor of solar cost trajectories [[44]](#ref-44).

### Current Cost Benchmarks

NREL ATB 2024 reports utility-scale solar at $24–42/MWh and onshore wind at $26–50/MWh, with 4-hour Li-ion storage at $115–165/MWh LCOS [[41]](#ref-41). Lazard LCOE+ v18.0 (June 2025) provides broader unsubsidized benchmarks: utility-scale solar at $38–78/MWh, onshore wind at $37–86/MWh, gas combined cycle at $48–109/MWh, geothermal at $66–109/MWh, and new-build nuclear at $141–220/MWh [[45]](#ref-45).

Regional variation is substantial: LBNL documents solar PPAs from $30–40/MWh (Southwest) to $60–90/MWh (Northeast) [[46]](#ref-46), and wind PPAs from $20–35/MWh (Great Plains) to $55–80/MWh (New England) [[47]](#ref-47). These differentials are central to the optimization — the cost-optimal mix varies significantly by region.

GridLab, Energy Futures Group, and Halcyon (September 2025) revealed CCGT projects routinely exceeding $2,000/kW — while the EIA AEO assumes only $921/kW and NREL ATB 2024 projects $1,638/kW. This massive disconnect between modeled assumptions and market reality has profound implications for capacity expansion planning [[48]](#ref-48).

### Enhanced Geothermal

Fervo Energy has emerged as the leading enhanced geothermal systems (EGS) developer. Cape Station in Utah — the world’s largest EGS development — targets 100 MW online by October 2026 and 500 MW by 2028, with drilling times fallen 70% year-over-year. DOE targets EGS capital cost of $3,700/kW by 2035 (from ~$28,000/kW in 2021) and LCOE of $45/MWh. CATF mapping suggests superhot rock geothermal could provide >4 TW of US capacity [[49]](#ref-49).

*Alignment with The 8,760 Problem:* The site’s four-pool supply model (Standard Supply Service / Corporate-Contracted / Merchant / New-Build) directly operationalizes the distinction between existing and new clean firm power. *Extension:* The critical mass analysis for Wright’s Law learning curve activation is novel — no published study quantifies how many corporate procurement decisions are needed to trigger specific learning rate milestones at the ISO level.

-----

## 5. Why LCOE Misleads: System Cost, Cannibalization, and Value Deflation

Joskow (2011, *American Economic Review*) established the foundational critique: LCOE is “seriously flawed” because it treats all MWh as homogeneous, overvaluing intermittent technologies relative to dispatchable ones [[50]](#ref-50). Hirth (2013, *Energy Economics*) quantified the consequence empirically, showing wind value factors drop to ~0.7 at 30% market share and solar value factors to ~0.7 at only 10–15% penetration [[51]](#ref-51). Hirth, Ueckerdt, and Edenhofer (2015, *Renewable Energy*) decomposed integration costs into profile costs (~25–35 €/MWh at 30–40% wind penetration, the dominant component), balancing costs (~2–4 €/MWh), and grid costs [[52]](#ref-52). Ueckerdt, Hirth, Luderer, and Edenhofer (2013, *Energy*) formalized System LCOE = Generation LCOE + Integration Costs, demonstrating that at moderate wind shares (~20%), integration costs can match generation costs [[53]](#ref-53).

The empirical reality of cannibalization is now starkly visible. López Prol, Steininger, and Zilberman (2020, *Energy Economics*) documented both absolute and relative cannibalization in CAISO [[54]](#ref-54). By 2024, solar capture rates in CAISO’s SP15 zone had plummeted to less than 30% (REsurety, 2025), meaning utility-scale solar earns 70% less revenue than a flat-output resource. Negative pricing in CAISO SP15 doubled from ~530 hours in 2023 to ~1,180 hours in 2024 (~13% of all hours), with median negative prices deepening from −$10 to −$17/MWh [[55]](#ref-55). CAISO data shows over 2.4 TWh of renewable curtailment in 2023, predominantly solar during midday hours [[56]](#ref-56). For hourly matching purchasers, curtailment represents a direct economic penalty: energy is procured and paid for but generates no matching credit because it exceeds instantaneous demand.

These dynamics are captured in NREL’s ReEDS modeling (Gagnon et al., 2024 Standard Scenarios) [[57]](#ref-57) and validated by the Princeton Net-Zero America Project’s finding that massive wind/solar/storage growth still requires significant roles for firm resources [[29]](#ref-29).

*Alignment with The 8,760 Problem:* The site’s synthetic LMP model (150,000+ scenarios per ISO) directly addresses the cannibalization challenge, and the curtailment cost quantification across matching targets and regions extends existing analysis. *Extension:* The stranded investment analysis — comparing interim vs. long-term target procurement decisions — is not found in the academic capacity expansion literature, which typically optimizes for a single future year.

-----

## 6. The Last Mile: Storage Economics, Grid Resilience, and Resource Adequacy

Battery storage enables temporal shifting of clean energy from surplus to deficit hours. The NREL ATB 2024 reports 4-hour lithium-ion round-trip efficiency of 85% [[41]](#ref-41), with installed costs declining from $380/kWh in 2020 to approximately $250/kWh in 2024 [[58]](#ref-58). BloombergNEF reports global average lithium-ion pack prices at $108/kWh in 2025 (down 93% since 2010), with stationary storage packs at $70/kWh [[59]](#ref-59). The EIA reports 16 GW of utility-scale battery storage installed in the U.S. as of 2024, with the majority in California (CAISO) and Texas (ERCOT) [[60]](#ref-60).

However, 4-hour storage has limited ability to address multi-day deficits or seasonal mismatches. Dowling et al. (2020) show that storage alone cannot cost-effectively replace firm generation at high decarbonization levels, as the required storage duration scales nonlinearly with the matching target [[61]](#ref-61). Cole et al. (2021, *Joule*, NREL) demonstrated that costs increase nonlinearly for the last few percent toward 100% renewable electricity [[62]](#ref-62). Mai et al. (2022, *Joule*, NREL) evaluated six strategies for the last 10%, concluding that no single strategy is sufficient [[63]](#ref-63).

### Long-Duration Energy Storage

Form Energy’s iron-air battery targets <$20/kWh for 100+ hours of continuous discharge, though current costs exceed $150/kWh [[64]](#ref-64). The DOE Long Duration Storage Shot (July 2021) targets 90% cost reduction for 10+ hour storage [[65]](#ref-65). The DOE Pathways to Commercial Liftoff for LDES (2023) estimates net-zero pathways deploying LDES yield $10–20 billion/year in savings and reduce the need for 200+ GW of new gas capacity [[66]](#ref-66). The LDES Council/McKinsey analysis (2021) concluded LDES is most competitive beyond 6–8 hours [[67]](#ref-67).

### Resource Adequacy and Dunkelflaute

PJM adopted marginal ELCC methodology (FERC-approved January 2024). Solar incremental capacity credit in CAISO is expected to drop to 6% by 2026 [[68]](#ref-68). Kittel et al. (2024, arXiv) quantified 2–10 Dunkelflaute events per year in northern Europe, mostly October–February. Germany experienced a major event in November 2024 with renewable contribution dropping to 30% and prices surging above €145/MWh [[69]](#ref-69).

### Critical Mineral Supply Chains

The supply chain for clean energy technologies faces material constraints. The IEA Critical Minerals Market Review 2024 documents concentration risk in key mineral supply chains [[70]](#ref-70). The USGS Mineral Commodity Summaries 2024 and DOE Critical Materials Assessment (2023) further detail bottlenecks in lithium, cobalt, nickel, and rare earths [[71]](#ref-71) [[72]](#ref-72). S&P Global’s copper analysis (2022) warns that demand from the energy transition may outstrip supply growth [[73]](#ref-73). Uranium supply remains adequate for current fleet operations, with the World Nuclear Association documenting stable mining production [[74]](#ref-74).

*Alignment with The 8,760 Problem:* The site’s treatment of the 90–100% zone with 2.5% granularity directly addresses the “last mile” problem. Its inclusion of diverse storage durations and firm power options mirrors the literature’s finding that no single technology solves the last 10%. *Extension:* The MAC vs. DAC crossover analysis with “no-regrets” resource identification represents a novel decision framework not found in the storage literature.

-----

## 7. The Interconnection Bottleneck and Market Design Challenges

Berkeley Lab’s “Queued Up” series (Rand et al., 2025) documents ~2,300 GW of capacity (10,300 projects) actively seeking grid connection. Only 13% of capacity from 2000–2019 reached commercial operations; 77% was withdrawn. Median wait times have doubled to over 4 years [[75]](#ref-75).

FERC Order No. 2023 (July 2023, with Order 2023-A in March 2024) represents the most significant interconnection reform in two decades, replacing first-come-first-served with cluster-based studies, increasing financial readiness requirements, and mandating interconnection heatmaps [[76]](#ref-76).

Wholesale market design faces fundamental challenges. Lo Prete, Palmer, and Robertson (RFF Report 24-09, June 2025) reviewed 11 proposed market designs against 10 criteria and concluded that existing designs are poorly suited for the evolving resource mix. Reliability challenges increasingly arise not from lack of capacity but from lack of available energy at times of need — a shift from capacity adequacy to energy adequacy [[77]](#ref-77).

PJM’s July 2024 capacity auction cleared at $269.92/MW-day, a nearly 10-fold increase from the prior year’s $28.92, driven by retirements, demand growth, and the new marginal ELCC methodology. ERCOT’s energy-only market faces separate challenges with loss-of-load expectation at 1.25 days/year — far above the 0.1 standard [[78]](#ref-78).

The “missing money” problem — where near-zero marginal cost renewables depress wholesale prices and erode revenue for resources needed for reliability — is worsening. Ela et al. (NREL, 2016) found net revenues insufficient to justify new thermal/nuclear investment in all base cases [[79]](#ref-79). Hogan (2017, *The Electricity Journal*) argued that out-of-market capacity mechanisms create “misallocated money” that undermines innovation [[80]](#ref-80).

The DOE National Transmission Needs Study (October 2023) concluded that interregional capacity must grow more than fivefold to realize IRA benefits [[81]](#ref-81). Simeone and Rose (NREL, June 2024) documented that at the ISO-NE/NYISO interface, power flowed in the economic direction for only 57% of hours in 2022 [[82]](#ref-82).

*Alignment with The 8,760 Problem:* The site covers all 7 US ISOs, inherently modeling differences between capacity markets (PJM, NYISO, ISO-NE), hybrid designs (MISO, SPP), and energy-only markets (ERCOT). *Extension:* No prior study systematically compares procurement optimization outcomes across all 7 ISOs simultaneously.

-----

## 8. Corporate Procurement Enters the Firm Power Era

BloombergNEF data shows corporations signed ~62 GW of clean power PPAs in 2024 before declining to 55.9 GW in 2025 — the first drop in nearly a decade. Meta, Amazon, Google, and Microsoft accounted for 49% of all global activity. Critically, 5.2 GW of 2025 deals were “baseload-like” products — a structural shift from the prior decade’s near-exclusive focus on wind and solar PPAs [[83]](#ref-83).

The nuclear deals are unprecedented: Microsoft signed a $1.6 billion PPA with Constellation for TMI Unit 1 restart (835 MW, September 2024). Google signed the first corporate SMR fleet deal with Kairos Power (up to 500 MW, October 2024) and committed capital to Elementl Power for 1.8 GW. Amazon is pursuing nearly 2 GW from Talen Energy’s Susquehanna plant. Meta signed 1.1 GW with Constellation’s Clinton Clean Energy Center (June 2025). Combined, Big Tech contracted for 10+ GW of possible new nuclear capacity in 2024–2025 [[84]](#ref-84) [[85]](#ref-85) [[86]](#ref-86) [[87]](#ref-87).

IRA Section 45V (final regulations January 2025) established hourly matching required starting January 1, 2030 for hydrogen production tax credits — the first federal hourly matching mandate in US law [[88]](#ref-88). The IEA World Energy Outlook 2024 provides the broader policy context [[89]](#ref-89), while Carbon Tracker’s stranded assets analysis demonstrates the financial risk of continued fossil fuel investment [[90]](#ref-90).

*Alignment with The 8,760 Problem:* The site contextualizes these corporate procurement decisions within its optimization framework, showing how different procurement strategies (annual vs. hourly, VRE-only vs. diversified) produce different system-level outcomes. *Extension:* The stranded investment analysis and Wright’s Law critical mass framework provide procurement strategists with quantitative tools for evaluating the system-level consequences of their individual decisions.

-----

## 9. Relationship to Capacity Expansion Models

The dominant tools for grid planning — GenX [[91]](#ref-91), ReEDS [[92]](#ref-92), SWITCH [[93]](#ref-93), and PyPSA [[94]](#ref-94) — are capacity expansion (CE) models. They solve a unified optimization: given cost assumptions, find the least-cost portfolio of generation, storage, and transmission that meets demand subject to reliability and policy constraints. They are powerful system-planning instruments. This optimizer is not one of them, and the distinction is deliberate.

**CE models answer a system planner’s question: “What should the grid build?”** They co-optimize capacity investment and operational dispatch within a single linear or mixed-integer program, typically over a 20–30 year horizon. To keep the problem computationally tractable, most CE models compress 8,760 hours into representative periods — often 50–200 time slices selected to preserve statistical properties of load and renewable output [[95]](#ref-95). This approximation is acceptable for system-level planning, where the goal is directionally correct capacity buildout under policy scenarios.

**This optimizer answers a procurement strategist’s question: “What does it cost to match my load cleanly at every hour, and what drives that cost?”** The unit of analysis is not the grid — it is a load-following clean energy portfolio evaluated against actual hourly demand and generation profiles. This reframing enables (and requires) a different methodological architecture.

### Full Temporal Resolution Without Representative-Period Compression

Hourly CFE matching is inherently a tail-risk problem. The hours that matter most — winter evening peaks, multi-day wind droughts, low-solar shoulder seasons — are precisely the hours that representative-period selection algorithms tend to underweight. A portfolio that scores 95% CFE on representative days may score 88% on the actual 8,760-hour profile because the hardest hours were averaged away. This optimizer simulates every hour of the year, which is essential for accurately sizing storage, quantifying curtailment losses, and identifying the resource mix transitions that occur at high matching thresholds.

### Physics-First Feasibility, Then Economic Evaluation

CE models couple physics and economics into a single objective: minimize cost subject to constraints. This produces one optimal point per scenario. If cost assumptions change, the entire optimization must re-run. This optimizer inverts the sequence. Step 1 exhaustively enumerates the *Physics Feasible Space* (PFS) — all resource mixes capable of achieving each hourly matching target from 50% to 100% — without reference to cost. Step 3 then evaluates the economic performance of every feasible mix under thousands of cost scenarios. Because physics and economics are decoupled, changing cost assumptions requires only a re-evaluation of the cached feasible space (minutes), not a re-solve of the physical dispatch (hours).

This architecture exposes structure that single-objective optimization conceals. The full PFS reveals that multiple qualitatively different portfolios can achieve the same matching target: a wind-heavy/low-storage mix and a solar-heavy/high-storage mix may both reach 90% CFE, but their cost sensitivity profiles diverge sharply under different technology cost futures. A CE model would return whichever is cheaper under its assumed inputs; this optimizer returns both and maps their relative economics across the entire sensitivity space. For a procurement strategist evaluating a 15–20 year PPA commitment under cost uncertainty, this visibility into portfolio robustness across scenarios is directly actionable.

### Parametric Sensitivity at Marginal Cost

The cached PFS of ~21.4 million physics-validated mixes enables what would be computationally prohibitive in a CE framework: exhaustive parametric sensitivity analysis across 5,832 cost scenarios per region and threshold (17,496 for CAISO). Each scenario varies renewable generation costs, firm generation costs, storage costs, fossil fuel prices, transmission costs, CCS technology costs, and 45Q policy assumptions simultaneously. In a CE model, each combination would require a full re-solve; here, it requires a vectorized cost evaluation on the pre-computed feasible set. This enables the ANOVA-style sensitivity decomposition — identifying which cost toggles dominate total cost variance at each matching threshold and in each region — which would be impractical to derive from a model requiring hours per solve.

### What This Approach Does Not Do

The tradeoffs are real. This optimizer does not model unit commitment, spinning reserves, or ramp-rate constraints — operational details that matter for grid reliability but are secondary for portfolio-level procurement analysis. It treats each ISO as a single node, omitting intra-regional transmission congestion. Storage dispatch follows a deterministic priority rule (short-duration batteries first, then long-duration) rather than being co-optimized with generation dispatch per cost scenario. And it is a snapshot model using 5-year averaged hourly profiles (2021–2025 element-wise averages for shapes, 2025 actuals for quantities), not a multi-decade capacity expansion pathway.

CO₂ emission rates are static annual averages per ISO and fuel type (sourced from EPA eGRID), applied uniformly across all 8,760 hours. The model does not capture hourly marginal emission rate variation — real grids exhibit significant diurnal and seasonal shifts in marginal fuel mix. Fuel-switching elasticity is also absent: the fossil fuel price toggle affects LMP-based cost calculations but does not alter emission factors. The deployment queue is an economic ordering (cheapest marginal abatement cost first, with within-ISO sequential buildout constraints), not a representation of physical FERC interconnection queue position, permitting timelines, or supply chain bottlenecks.

These are appropriate simplifications for the question being asked. A corporate buyer evaluating hourly CFE procurement does not need to solve the system planner’s problem. They need to understand how their portfolio’s cost and physical performance interact across plausible futures — and that requires full temporal resolution, exhaustive feasibility mapping, and rapid parametric evaluation, none of which are strengths of the CE paradigm.

The two approaches are complementary. CE models like GenX and ReEDS generate the technology cost projections and system-level deployment trajectories that inform this optimizer’s input assumptions (LCOE ranges, transmission costs, capacity factors). This optimizer then applies those assumptions to the specific problem of hourly load matching — a procurement-scale question that system models are not designed to answer at the temporal granularity required.

-----

## 10. Novel Contributions of The 8,760 Problem

Across all research domains reviewed, the following nine contributions are not found in any prior published work:

1. **Co-optimization of cost and resource mix across 5,832 sensitivity scenarios per ISO** (vs. single-point estimates in existing models). The parametric breadth enables ANOVA-style sensitivity decomposition identifying which cost toggles dominate variance at each threshold and region.
1. **The four-pool supply model** (Standard Supply Service / Corporate-Contracted / Merchant / New-Build). This operationalizes the distinction between existing and new clean resources in a way that reflects actual procurement market structure.
1. **The 10-strategy procurement comparison framework** (3 consequential, 3 hourly, 4 annual). The most comprehensive side-by-side evaluation of procurement strategies documented to date — existing studies compare at most 2–3 strategies.
1. **The critical mass analysis for Wright’s Law learning curve activation.** No published study quantifies how many corporate procurement decisions are needed to trigger specific learning rate milestones for advanced nuclear or enhanced geothermal at the ISO level.
1. **The ~33,000 physics-feasible resource mixes per ISO** (21.4 million total). The exhaustive enumeration of the Physics Feasible Space exposes structure that single-objective optimization conceals.
1. **The stranded investment analysis** (interim vs. long-term target comparison). Not found in the academic capacity expansion literature, which typically optimizes for a single future year.
1. **The synthetic LMP model** (150,000+ scenarios per ISO). Extends beyond existing temporal mismatch analysis to generate comprehensive price distributions.
1. **MAC vs. DAC crossover analysis.** Identifies where hourly matching becomes more expensive than direct air capture, with “no-regrets” resource identification below the threshold.
1. **2.5% threshold granularity in the 90–100% inflection zone.** Finer than any published study, providing practitioners the precision needed for target-setting in the most cost-sensitive zone.

-----

## Conclusion

Across all seven domains, the research literature converges: annual REC matching has negligible system-level emissions impact; hourly matching with diverse clean resources is essential but costly above ~90%; firm clean power reduces system costs by tens of percent; LCOE misleads about dispatchable resource value; interconnection queues create binding constraints; and corporate procurement is shifting toward firm, hourly-matched products.

What the literature lacks — and what The 8,760 Problem provides — is a unified analytical framework connecting these findings across all seven domains simultaneously, at the geographic specificity of individual ISOs, with the scenario breadth needed to support actual procurement decisions under deep uncertainty. By bridging academic models and practitioner needs with unprecedented scenario scale, the site occupies a genuinely novel position — one that the imminent GHG Protocol Scope 2 revision, SBTi V2 requirements, and the growth of corporate firm-power procurement will make increasingly relevant through 2027 and beyond.

-----

## References

### CO₂ Accounting & Frameworks

<a id="ref-1"></a>1. World Resources Institute. (2015). “GHG Protocol Scope 2 Guidance: An amendment to the GHG Protocol Corporate Standard.” WRI. <https://ghgprotocol.org/scope-2-guidance>

<a id="ref-2"></a>2. Xu, Q., et al. (2024). “Temporal Granularity in Clean Energy Accounting: Implications of Annual vs. Hourly Matching for Grid Decarbonization.” *Energy Policy*, 184, 113892.

<a id="ref-3"></a>3. Bjørn, A., Lloyd, S., Brander, M., Matthews, H.D. (2022). “Renewable energy certificates threaten the integrity of corporate science-based targets.” *Nature Climate Change*, 12, 539–546.

<a id="ref-4"></a>4. GHG Protocol. (2024–2026). Scope 2 Guidance Standard Development Plan; Phase 1 public consultation (October 2025 – January 2026). <https://ghgprotocol.org>

<a id="ref-5"></a>5. Brander, M. (2022). “The most important GHG accounting concept you may not have heard of: the attributional–consequential distinction.” *Carbon Management*, 13(1), 525–538.

<a id="ref-6"></a>6. Brander, M. & Bjørn, A. (2023). “Attributional and consequential approaches to GHG accounting.” *International Journal of Life Cycle Assessment*, 28, 1–13.

<a id="ref-7"></a>7. Bjørn, A., et al. (2024). “Ensuring low-emission electricity purchasing requires broader systems perspective.” *Nature Communications*, 15, 4518.

### 24/7 CFE Programs & Compacts

<a id="ref-8"></a>8. Google. (2023–2024). “24/7 Carbon-Free Energy: Methodology and Metrics.” Google Sustainability Reports. <https://sustainability.google/progress/energy/>

<a id="ref-9"></a>9. United Nations Energy. (2021). “24/7 Carbon-free Energy Compact.” <https://www.un.org/en/energy-compacts/page/compact-247>

### Marginal Emissions & Data

<a id="ref-10"></a>10. Siler-Evans, K., Azevedo, I.L., Morgan, M.G. (2012). “Marginal Emissions Factors for the U.S. Electricity System.” *Environmental Science & Technology*, 46(9), 4742–4748.

<a id="ref-11"></a>11. VERACI-T Working Group. (2023). “Towards Objective Evaluation of the Accuracy of Marginal Emission Factor Models.”

### Granular Certificates & Standards

<a id="ref-12"></a>12. EnergyTag. (2024–2025). Granular Certificate Scheme Standard V2 (December 2024); Matching Standard V1 (March 2024). <https://energytag.org>

<a id="ref-13"></a>13. Science Based Targets initiative. (2025). “SBTi Corporate Net-Zero Standard V2.0.” Second Consultation Draft (November 2025).

<a id="ref-14"></a>14. Science Based Targets initiative. (2025). “SBTi Power Sector Net-Zero Standard.” First Draft (September 2025).

<a id="ref-15"></a>15. EPRI. (2023–2025). “SMARTargets Framework.” Announced COP28 December 2023; public consultation closed August 2025.

<a id="ref-16"></a>16. Ceres. (2025). “Ceres calls on EPRI to strengthen their draft emissions reduction guidance.” (August 2025).

### Hourly CFE Modeling

<a id="ref-17"></a>17. Xu, Q., Ricks, W., Manocha, A., Patankar, N., Jenkins, J.D. (2024). “System-level Impacts of Voluntary Carbon-free Electricity Procurement.” *Joule*, 8(1), 200–219.

<a id="ref-18"></a>18. Xu, Q., et al. (2024). “System-level impacts of 24/7 carbon-free energy procurement” [supply stack competition analysis]. *Joule*.

<a id="ref-19"></a>19. Riepin, I. & Brown, T. (2024). “Hourly matching of electricity supply and demand for large buyers.” *Energy Strategy Reviews*, 52, 101337.

<a id="ref-20"></a>20. Riepin, I., Jenkins, J.D., Swezey, B., Brown, T. (2025). “24/7 CFE matching and technology learning.” *Joule* [commentary].

<a id="ref-21"></a>21. Dyson, M., Shah, M., Teplin, C. (2021). “Clean Power by the Hour.” Rocky Mountain Institute (RMI).

<a id="ref-22"></a>22. Miller, G., Lott, M.C., et al. (2022). “24/7 Carbon-Free Energy: Methodologies and Case Studies.” International Energy Agency (IEA). <https://www.iea.org/reports/247-carbon-free-energy>

<a id="ref-23"></a>23. TransitionZero. (2025). “24/7 CFE Modeling: India, Japan, Singapore, Malaysia.” Series funded by Google.org.

<a id="ref-24"></a>24. Denholm, P., et al. (2022). “Examining Supply-Side Options to Achieve 100% Clean Electricity by 2035.” NREL/TP-6A20-81644. <https://www.nrel.gov/docs/fy22osti/81644.pdf>

### EAC Markets & Voluntary Procurement

<a id="ref-25"></a>25. Gillenwater, M. (2008). “Redefining RECs—Part 1: Untangling Attributes and Offsets.” *Energy Policy*, 36(6), 2109–2119.

<a id="ref-26"></a>26. O’Shaughnessy, E., Jena, S., Sumner, J. (2025). “Status and Trends in the U.S. Voluntary Green Power Market (2023 Data).” NREL/TP-6A20-92289.

### Clean Firm Power Economics

<a id="ref-27"></a>27. Sepulveda, N.A., Jenkins, J.D., de Sisternes, F.J., Lester, R.K. (2018). “The Role of Firm Low-Carbon Electricity Resources in Deep Decarbonization of Power Generation.” *Joule*, 2(12), 2403–2420.

<a id="ref-28"></a>28. Long, J.C.S., et al. (2021). “Clean Firm Power is the Key to California’s Carbon-Free Energy Future.” *Issues in Science and Technology* [EDF/CATF California study].

<a id="ref-29"></a>29. Larson, E., et al. (2021). “Net-Zero America: Potential Pathways, Infrastructure, and Impacts.” Princeton University. <https://netzeroamerica.princeton.edu/>

<a id="ref-30"></a>30. Jenkins, J.D., Luke, M., Thernstrom, S. (2018). “Getting to Zero Carbon Emissions in the Electric Power Sector.” *Joule*, 2(12), 2498–2510.

<a id="ref-31"></a>31. Clean Air Task Force. (2024–2026). “Clean Firm Electricity Technologies: What, Why, How” (Spokas & Ricks, February 2026); “Beyond LCOE” (June 2025). CATF. <https://www.catf.us/clean-firm-electricity/>

### Nuclear Economics

<a id="ref-32"></a>32. U.S. Energy Information Administration. (2024). “Nuclear power plant operations and technology.” EIA. <https://www.eia.gov/nuclear/>

<a id="ref-33"></a>33. U.S. Department of Energy. (2024). “Pathways to Commercial Liftoff: Advanced Nuclear.” (September 2024 update). <https://liftoff.energy.gov/advanced-nuclear/>

<a id="ref-34"></a>34. Shirvan, K. (2024). MIT CANES Report ANP-201 TR. MIT Advanced Nuclear Power Program.

<a id="ref-35"></a>35. Bolisetti, A., et al. (2024). “Quantifying Capital Cost Reduction Pathways for Advanced Nuclear.” INL/RPT-24-7767. Idaho National Laboratory.

<a id="ref-36"></a>36. Abou-Jaoude, A., et al. (2024). “Meta-Analysis of Advanced Nuclear Reactor Cost Estimations.” INL/RPT-23-72972. Idaho National Laboratory.

<a id="ref-37"></a>37. NREL. (2024). “Annual Technology Baseline (ATB) 2024: Nuclear Module.” NREL. <https://atb.nrel.gov/electricity/2024/technologies>

<a id="ref-38"></a>38. IEEFA. (2023). “Eye-Popping New Cost Estimates Released for NuScale Small Modular Reactor.” [~$20,000/kW at cancellation; $89/MWh final LCOE estimate].

<a id="ref-39"></a>39. Georgia Power / Southern Company. (2023). Vogtle Units 3&4 construction completion — realized cost ~$15,000/kW for 2.2 GW AP1000 expansion.

<a id="ref-40"></a>40. Frontiers in Energy Research. (2022). “Cost Projection of Natural Gas Combined Cycle with Post-Combustion Carbon Capture.” [FOAK ~$103/MWh; NOAK ~$92/MWh].

### Cost Benchmarks

<a id="ref-41"></a>41. NREL. (2024). “Annual Technology Baseline (ATB) 2024: Electricity.” National Renewable Energy Laboratory. <https://atb.nrel.gov/electricity/2024/technologies>

<a id="ref-42"></a>42. Wright, T.P. (1936). “Factors Affecting the Cost of Airplanes.” *Journal of the Aeronautical Sciences*, 3(4), 122–128.

<a id="ref-43"></a>43. Bolinger, M., Wiser, R., O’Shaughnessy, E. (2022). “Levelized cost-based learning analysis of utility-scale wind and solar in the United States.” *iScience*, 25(6), 104378.

<a id="ref-44"></a>44. Way, R., Iribarren, M.C., Hepburn, C., Farmer, J.D. (2022). “Empirically grounded technology forecasts and the energy transition.” *Joule*, 6(9), 2057–2082.

<a id="ref-45"></a>45. Lazard. (2025). “Lazard’s Levelized Cost of Energy Analysis, Version 18.0.” <https://www.lazard.com/research-insights/levelized-cost-of-energyplus/>

<a id="ref-46"></a>46. Bolinger, M., Seel, J., et al. (2024). “Utility-Scale Solar, 2024 Edition.” Lawrence Berkeley National Laboratory. <https://emp.lbl.gov/utility-scale-solar>

<a id="ref-47"></a>47. Wiser, R., Bolinger, M., et al. (2024). “Land-Based Wind Market Report: 2024 Edition.” Lawrence Berkeley National Laboratory. <https://emp.lbl.gov/wind-technologies-market-report>

<a id="ref-48"></a>48. GridLab, Energy Futures Group, and Halcyon. (2025). “Gas Turbine Cost Report.” (September 2025).

### Enhanced Geothermal

<a id="ref-49"></a>49. Fervo Energy. (2024–2025). Cape Station development updates; DOE Enhanced Geothermal Shot targets; CATF Superhot Rock assessment.

### System Cost & Value Deflation

<a id="ref-50"></a>50. Joskow, P.L. (2011). “Comparing the Costs of Intermittent and Dispatchable Electricity Generating Technologies.” *American Economic Review*, 101(3), 238–241.

<a id="ref-51"></a>51. Hirth, L. (2013). “The Market Value of Variable Renewables: The Effect of Solar Wind Power Variability on their Relative Price.” *Energy Economics*, 38, 218–236.

<a id="ref-52"></a>52. Hirth, L., Ueckerdt, F., Edenhofer, O. (2015). “Integration costs revisited — An economic framework for wind and solar variability.” *Renewable Energy*, 74, 925–939.

<a id="ref-53"></a>53. Ueckerdt, F., Hirth, L., Luderer, G., Edenhofer, O. (2013). “System LCOE: What are the costs of variable renewables?” *Energy*, 63, 61–75.

<a id="ref-54"></a>54. López Prol, J., Steininger, K.W., Zilberman, D. (2020). “The cannibalization effect of wind and solar in the California wholesale electricity market.” *Energy Economics*, 85, 104552.

<a id="ref-55"></a>55. REsurety / Bhandari, N. (2025). “Negative Prices in CAISO.” (April 2025). <https://resurety.com/article-negative-prices-in-caiso/>

<a id="ref-56"></a>56. California ISO (CAISO). (2024). “Managing Oversupply.” CAISO Market Performance Reports. <https://www.caiso.com/informed/Pages/ManagingOversupply.aspx>

### Capacity Expansion Models & Standard Scenarios

<a id="ref-57"></a>57. Gagnon, P., et al. (2024). “2024 Standard Scenarios Report.” NREL/TP-6A40-92256. National Renewable Energy Laboratory.

### Storage & Grid Resilience

<a id="ref-58"></a>58. Feldman, D., et al. (2024). “U.S. Solar Photovoltaic System and Energy Storage Cost Benchmarks, With Minimum Sustainable Price Analysis: Q1 2024.” NREL. <https://www.nrel.gov/docs/fy24osti/88554.pdf>

<a id="ref-59"></a>59. BloombergNEF. (2025). “Battery Price Survey 2025.”

<a id="ref-60"></a>60. U.S. Energy Information Administration. (2024). “Battery Storage in the United States: An Update on Market Trends.” EIA. <https://www.eia.gov/analysis/studies/electricity/batterystorage/>

<a id="ref-61"></a>61. Dowling, J.A., et al. (2020). “Role of Long-Duration Energy Storage in Variable Renewable Electricity Systems.” *Joule*, 4(9), 1907–1928.

<a id="ref-62"></a>62. Cole, W.J., et al. (2021). “Quantifying the challenge of reaching a 100% renewable energy power system for the United States.” *Joule*, 5(7), 1732–1748.

<a id="ref-63"></a>63. Mai, T., et al. (2022). “On the path to 100% clean electricity: six strategies for the last 10%.” *Joule*, 6(11), 2521–2536.

<a id="ref-64"></a>64. Form Energy. (2024–2025). Iron-air battery technology and deployment pipeline.

<a id="ref-65"></a>65. U.S. Department of Energy. (2021). “Long Duration Storage Shot.” (July 2021).

<a id="ref-66"></a>66. U.S. Department of Energy. (2023). “Pathways to Commercial Liftoff: Long Duration Energy Storage.”

<a id="ref-67"></a>67. Long Duration Energy Storage Council / McKinsey. (2021). “Net-zero power: Long duration energy storage for a renewable grid.”

<a id="ref-68"></a>68. PJM. (2024). Marginal ELCC methodology (FERC-approved January 2024).

<a id="ref-69"></a>69. Kittel, M., et al. (2024). “Quantifying the Dunkelflaute.” arXiv preprint.

### Critical Minerals & Supply Chains

<a id="ref-70"></a>70. International Energy Agency. (2024). “Critical Minerals Market Review 2024.” IEA. <https://www.iea.org/reports/critical-minerals-market-review-2024>

<a id="ref-71"></a>71. U.S. Geological Survey. (2024). “Mineral Commodity Summaries 2024.” USGS. <https://www.usgs.gov/centers/national-minerals-information-center/mineral-commodity-summaries>

<a id="ref-72"></a>72. U.S. Department of Energy. (2023). “Critical Materials Assessment.” DOE. <https://www.energy.gov/sites/default/files/2023-07/doe-critical-material-assessment_07312023.pdf>

<a id="ref-73"></a>73. S&P Global. (2022). “The Future of Copper: Will the Looming Supply Gap Short-Circuit the Energy Transition?” S&P Global Market Intelligence. <https://www.spglobal.com/marketintelligence/en/mi/research-analysis/the-future-of-copper.html>

<a id="ref-74"></a>74. World Nuclear Association. (2024). “World Uranium Mining Production.” <https://world-nuclear.org/information-library/nuclear-fuel-cycle/uranium-resources/supply-of-uranium>; see also U.S. Energy Information Administration, “Uranium Marketing Annual Report” (2024). <https://www.eia.gov/uranium/marketing/>

### Interconnection & Market Design

<a id="ref-75"></a>75. Rand, J., et al. (2025). “Queued Up: Characteristics of Power Plants Seeking Transmission Interconnection.” Lawrence Berkeley National Laboratory. <https://emp.lbl.gov/queues>

<a id="ref-76"></a>76. Federal Energy Regulatory Commission. (2023–2024). Order No. 2023 (July 2023) and Order 2023-A (March 2024).

<a id="ref-77"></a>77. Lo Prete, C., Palmer, K., Robertson, D. (2025). “Time for a Market Upgrade?” RFF Report 24-09. Resources for the Future. (June 2025).

<a id="ref-78"></a>78. FERC / ISO Market Monitoring Reports (2023–2024). PJM Independent Market Monitor State of the Market Report; NYISO, ISO-NE, CAISO, ERCOT market monitoring reports.

<a id="ref-79"></a>79. Ela, E., et al. (2016). “Revenue sufficiency and reliability in a zero-marginal-cost future.” NREL.

<a id="ref-80"></a>80. Hogan, W.W. (2017). “Electricity Market Design and the Green Agenda.” *The Electricity Journal*, 30(8), 26–42.

<a id="ref-81"></a>81. U.S. Department of Energy. (2023). “National Transmission Needs Study.” (October 2023).

<a id="ref-82"></a>82. Simeone, C. & Rose, K. (2024). “Inter-regional transfer analysis.” NREL. (June 2024).

### Corporate Procurement & Policy

<a id="ref-83"></a>83. BloombergNEF. (2024–2025). “Corporate Energy Market Outlook” and corporate PPA data.

<a id="ref-84"></a>84. Microsoft / Constellation. (2024). Three Mile Island Unit 1 restart PPA. (September 2024).

<a id="ref-85"></a>85. Google / Kairos Power. (2024). SMR fleet deal. (October 2024); Google / Elementl Power (May 2025).

<a id="ref-86"></a>86. Amazon / X-energy / Talen Energy. (2024–2025). Nuclear commitments.

<a id="ref-87"></a>87. Meta / Constellation. (2025). Clinton Clean Energy Center deal. (June 2025).

<a id="ref-88"></a>88. IRA Section 45V. (2025). Final Regulations (January 2025). Hourly matching required starting January 1, 2030.

<a id="ref-89"></a>89. International Energy Agency. (2024). “World Energy Outlook 2024.” IEA.

<a id="ref-90"></a>90. Carbon Tracker Initiative. (2023–2024). Stranded Assets analysis.

### Capacity Expansion Models

<a id="ref-91"></a>91. Jenkins, J.D., Sepulveda, N.A., et al. (2017–2024). “GenX: Configurable Capacity Expansion Model.” MIT Energy Initiative / Princeton ZERO Lab. <https://github.com/GenXProject/GenX>

<a id="ref-92"></a>92. Ho, J., et al. (2021). “Regional Energy Deployment System (ReEDS) Model Documentation: Version 2020.” NREL/TP-6A20-78195. <https://www.nrel.gov/analysis/reeds/>

<a id="ref-93"></a>93. Johnston, J., et al. (2019). “Switch 2.0: A Modern Platform for Planning High-Renewable Power Systems.” *SoftwareX*, 10, 100251.

<a id="ref-94"></a>94. Brown, T., Hörsch, J., Schlachtberger, D. (2018). “PyPSA: Python for Power System Analysis.” *Journal of Open Research Software*, 6(4). <https://pypsa.org/>

<a id="ref-95"></a>95. Kotzur, L., et al. (2018). “Impact of Different Time Series Aggregation Methods on Optimal Energy System Design.” *Renewable Energy*, 117, 474–487.

### Additional Data Sources

<a id="ref-96"></a>96. U.S. Energy Information Administration. (2025). “Hourly Electric Grid Monitor (EIA-930).” EIA. <https://www.eia.gov/electricity/gridmonitor/>

<a id="ref-97"></a>97. Miller, G., et al. (2024). “Cost and Emissions Impact of Voluntary Clean Energy Procurement Strategies.” *The Electricity Journal*, 37(2), 107371.

<a id="ref-98"></a>98. DOE Office of Clean Energy Demonstrations (OCED). (2024). “Portfolio Insights: Carbon Capture in the Power Sector.” [FOAK→NOAK ~25% CAPEX reduction for CCS].

<a id="ref-99"></a>99. NETL. (2023). “Baseline Study: High Carbon Capture Rates.” [CCS learning rates: CO₂ capture 3%/doubling; transport & sequestration 5%/doubling].

<a id="ref-100"></a>100. LDES Council. (2023). “The Journey to Net-Zero: An Update on Long Duration Energy Storage.” <https://www.ldescouncil.com/insights>