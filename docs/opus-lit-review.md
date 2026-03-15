# Peer review and audit of “The 8,760 Problem”

**“The 8,760 Problem” presents a genuinely novel contribution to clean energy procurement analysis — its exhaustive enumeration of ~33,000 resource mixes across 5,832+ cost scenarios per ISO provides a transparency and uncertainty coverage that no major capacity expansion model matches.** But this strength coexists with methodological choices that systematically shape its findings in ways the site does not fully disclose. This review identifies seven categories of vulnerability, ranging from foundational modeling assumptions to policy risks that have already materialized since the site’s 2025 baseline was established. The core concern is not that the analysis is wrong, but that its boundary conditions — single weather year, independent ISOs, no transmission, static snapshot, reliance on learning curves with negative historical precedent in nuclear — bound the solution space in ways that could materially alter its conclusions.

-----

## 1. The single weather year creates a ±10% blind spot in system cost

The site’s 8,760-hour dispatch optimization uses a single year (2025) of hourly generation profiles. The academic literature establishes this as a significant vulnerability. Gøtske, Andresen, Neumann, and Victoria (2024) in *Nature Communications* optimized a European energy system across **62 different weather years** (1960–2021) and found **±10% variation in total system costs** depending on which year was selected. The distribution exhibits long tails driven by compound weather events   — precisely the scenarios that stress-test clean firm portfolios. More critically, Schill and Zerrahn (2022) in *Environmental Research Letters* demonstrated that single-year optimizations **underestimate storage energy requirements by more than half** compared to multi-year analysis using 35 years of German reanalysis data.  A 2025 ScienceDirect study running 20 weather-year simulations for ERCOT confirmed that while aggregate capacity projections are somewhat robust, **regional capacity allocation, cumulative costs, and emissions vary substantially** across weather years. 

The site’s ~33,000 resource mixes are evaluated against one weather realization. Portfolios optimized for an unusually sunny or windy year may perform poorly in a low-resource year. This particularly affects storage-dependent portfolios and the relative competitiveness of clean firm power: in a high-variability weather year, firm resources appear more valuable; in a benign year, they appear unnecessarily expensive. **The site should acknowledge this limitation explicitly and ideally present sensitivity to at least 3–5 weather years** drawn from the recent climatological record, including one drought/low-wind year and one extreme weather year.

-----

## 2. Enumeration finds the best of 33,000 mixes — but the true optimum may lie between them

The site’s approach of enumerating ~33,000 “physics-feasible” resource mixes and running full 8,760-hour dispatch for each is conceptually a screening/enumeration methodology. This has important implications relative to true capacity expansion optimization as implemented in models like GenX, PyPSA, ReEDS, and US-REGEN.

**What the site gains from enumeration is real.** Full visibility into the solution space is valuable — Neumann and Brown (2021) showed that even a 0.5% cost deviation from optimal in European system models reveals vast diversity in feasible investment portfolios.  The site’s approach avoids solver artifacts, time-domain reduction approximations, and the “black box” problem of mathematical optimization. Its coverage of **5,832+ cost scenarios × 150,000+ LMP scenarios** far exceeds what any capacity expansion model typically evaluates. Decision-makers can explore trade-offs directly rather than receiving a prescriptive “optimal” answer.

**What the site loses is also real.** Jenkins and Sepulveda (2017) in the GenX documentation explicitly critique screening approaches for ignoring inter-temporal constraints on ramp rates, unit commitment decisions (startup costs, minimum stable output), operating reserves, and the co-optimization of investment and operations.   Palmintier and Webster (2011) demonstrated that including unit-commitment details **significantly changes the optimal capacity mix**  compared to traditional screening methods. Most critically, Sepulveda et al. (2018) found that installed capacity of some resources changes **non-monotonically** as emissions limits tighten  — patterns discoverable only through co-optimization, not grid enumeration. The discrete step sizes between the site’s enumerated mixes determine whether near-optimal solutions are captured; the true global optimum across continuous capacity combinations may fall between grid points.

**The site should be transparent that its ~33,000 mixes represent a discretized sample of the feasible space, not the complete solution landscape.** It should disclose the granularity of its enumeration grid (e.g., capacity step sizes for each technology) so reviewers can assess whether key regions of the solution space are adequately covered.

-----

## 3. Modeling ISOs independently ignores the grid’s most powerful cost lever

Perhaps the most consequential methodological limitation is modeling each of 7 ISOs independently without inter-regional power flows. Brown and Botterud (2021) in *Joule* demonstrated that **inter-state coordination and transmission expansion reduce system cost by 46%** in a 100%-renewable US system compared to a state-by-state approach.  The DOE National Transmission Planning Study (2024) found that transmission expansion lowers US electric system costs by **$270–490 billion through 2050** in low-carbon scenarios.  CAISO alone imports approximately **26% of its electricity** from other states  — modeling it as an island fundamentally misrepresents its resource economics.

Safavi, Kemp, Gorman, Millstein, and Wiser (2026) analyzed 32 inter-ISO interfaces over 2014–2023 and found average annual economic gains from inter-regional flow of **$1.23 billion per year**,  with many interfaces underutilized even during high-value periods. The implication is stark: optimal portfolios within each ISO are strongly influenced by available imports and exports. A portfolio that appears expensive in isolation — because it must self-provide all nighttime clean energy — may be significantly cheaper when it can import hydropower from the Pacific Northwest or wind from the Great Plains. **Independent ISO modeling systematically overstates the cost of hourly matching and understates the value of geographic resource diversity.** This bias likely inflates the apparent value of clean firm power (which reduces import dependence) relative to inter-regional renewable portfolios.

-----

## 4. Missing demand flexibility worth $15 billion per year reshapes optimal portfolios

The site treats load as perfectly inelastic across all 8,760 hours. The Brattle Group (2019) estimated that current US demand response capability of ~60 GW could nearly double to ~120 GW, with national benefits exceeding **$15 billion per year** by 2030. RMI found that flexible demand can **avoid almost $2 billion in annual generator costs** and reduce renewable curtailment by 40% in modeled systems.  NREL analysis showed demand response provides flexibility equivalent to **1 GW of 6-hour battery storage** across the Florida Reliability Coordinating region. 

By ignoring demand flexibility, the site overestimates the need for supply-side resources — particularly peaking capacity and storage — and underestimates the role of load shifting in achieving hourly matching at lower cost. With DR potentially representing **10–20% of peak demand**  and providing storage-equivalent services, optimal portfolios could shift meaningfully. The site also ignores demand growth from data centers, EVs, and building electrification — projected to be the largest US demand increase since WWII  — meaning the 2025 load profile may not represent the system these portfolios are designed to serve.

-----

## 5. The clean firm power case rests on learning curves that nuclear has never achieved

The site’s advocacy for clean firm power (nuclear, geothermal, long-duration storage) draws on modeling that assumes these technologies reach competitive costs through experience-curve learning. The literature raises profound concerns about applying Wright’s Law to nuclear power.

**Grubler (2010) in *Energy Policy*** documented that even the French nuclear program — “arguably the most successful nuclear scale-up in an industrialized country” with centralized decision-making, regulatory stability, and standardized designs — exhibited **substantial cost escalation**.  Units completed after 1990 were **3.5× more costly** than 1970s reactors. Grubler concluded: “Not only do nuclear reactors across all countries with significant programs invariably exhibit negative learning, but the pattern is also quite variable, defying approximations by simple learning-curve models.”  Koomey, Hultman, and Grubler (2017) rebutted the more optimistic reading of Lovering, Yip, and Nordhaus (2016), arguing the latter cherry-picked data and excluded interest costs that substantially undercount true escalation. 

**Nordhaus (2014) in *The Energy Journal*** raised a fundamental identification problem: it is statistically impossible to reliably separate learning-by-doing from exogenous technological change. Estimated learning coefficients are **“generally biased upwards,”**  and this bias can cause “massive misallocation of resources” in optimization models.  Rubin et al. (2015) in *Energy Policy* found **“substantial variability (as much as an order of magnitude)”** in reported learning rates across studies for all 11 technologies examined.  A 2025 study in *ScienceDirect* analyzing 87 technologies found that **“past learning rates are not good predictors of future learning rates”** — with stepwise changes providing a better fit than constant rates for 58 of 87 technologies.

The real-world evidence is sobering. NuScale’s UAMPS project saw costs escalate from **$5.3 billion to $9.3 billion**  (75% increase) before cancellation. Sovacool et al. (2014) found **175 of 180 nuclear projects** worldwide exceeded budget by an average of **117%**. Vogtle Units 3&4 cost **$30–36.8 billion** versus the original $14 billion estimate.  Flamanville 3 reached **~€13.2 billion** versus €3.3 billion projected.  Hinkley Point C is now estimated at **£35 billion** (2015 prices) versus £18 billion original.  

The ITIF (2025) report “Small Modular Reactors: A Realist Approach” — while generally supportive of nuclear — acknowledged that first-of-a-kind SMRs **“will likely cost more per MWh than existing large reactors, and certainly more than competing fuels,”** and that “we don’t yet know whether SMRs will crack the scale-up problem; that question cannot be answered for at least a decade.”  Way et al. (2022) in *Joule* found nuclear, hydropower, and biopower show **“flat or rising costs,”** concluding these technologies “have less potential to play a significant role in energy transition”  from a learning-curve perspective. **A ±5 percentage point change in assumed learning rate produces approximately a 2× difference in projected costs after 10 doublings** — meaning the site’s conclusions are acutely sensitive to this assumption.

-----

## 6. The hourly-vs-annual debate is more contested than the site may acknowledge

The strongest counterargument to hourly matching comes from WattTime’s 2025 analysis, which found that for a fixed budget of $4.6 billion, impact accounting (emissions-first, location-flexible procurement) avoids **211 MT CO₂/year** versus only **45 MT CO₂/year** for 98% hourly matching — a **4.7× advantage**.   WattTime estimates that mandating hourly matching would **increase grid emissions by 42.6 MT CO₂e/year** because cost premiums drive companies out of voluntary procurement entirely,  based on demand elasticities of 0.5–0.96 documented in comparable markets. 

However, this argument faces its own challenges. Xu, Ricks, and Jenkins (2024) in *Joule* found that **temporal/hourly matching was the only procurement strategy that consistently lowered system-wide long-run emissions** in the US. Annual volumetric matching was “almost entirely ineffective,”   and emissions matching also failed because it doesn’t account for counterfactual deployment that would have occurred anyway.  The system-level impact — driving early adoption of advanced clean technologies  — may outweigh the per-dollar efficiency disadvantage.

The site should acknowledge three critical complications in this debate:

- **The non-additivity problem.** Bjørn, Lloyd, Brander, and Matthews (2022) in *Nature Climate Change* found companies using RECs reported  a **31% reduction** in Scope 2 emissions when actual reductions were only **~10%** — a roughly **3× overstatement**.  Brander and Bjørn (2023) demonstrated that marginal emission factors “are not compatible with allocational physical GHG accounting” because the property of additivity is violated  when all actors claim the same marginal impact. 
- **The collective action tension.** Riepin and Brown (2024) found that 24/7 CFE procurement reduces the need for flexibility in background electricity systems  — a positive externality participating buyers don’t capture. But if hourly matching costs drive participation from ~10% of C&I load to much lower levels, the total system-level benefit could be negative. 
- **The GHG Protocol Scope 2 revision.** The ISB voted 10-1 in July 2025 to advance hourly and regional matching for market-based reporting,  with the final standard expected in 2027.  But the ISB also voted 7-4 **against** advancing the Marginal Impact Method under Scope 2, redirecting consequential accounting to a separate supplemental framework.  Nearly **80% of surveyed companies** lack confidence they can procure time-matched clean electricity within smaller market boundaries.  

-----

## 7. The policy ground has shifted beneath the site’s 2025 snapshot

The most urgent vulnerability is that the policy environment assumed by the site has already changed fundamentally. **The One Big Beautiful Bill Act (OBBBA), signed July 4, 2025**, enacted an accelerated phaseout of IRA clean energy tax credits with a hard deadline: wind and solar projects must begin construction before **July 5, 2026** or be placed in service by **December 31, 2027** to claim ITC/PTC credits.   Battery storage and nuclear retain credit access through 2032+, but wind and solar credits face  what industry has called the “July 4, 2026 cliff.” 

Rhodium Group analysis projects this will **shrink new clean capacity additions by 53–59% through 2035** relative to baseline,   with **$522 billion** of clean energy investment at risk.   GHG emissions are projected to be **315–574 million metric tons higher** in 2035 than pre-OBBBA baseline.  The economics of wind and solar have fundamentally changed for projects not already under construction, shifting the cost calculus toward battery storage and nuclear — which retain credits — in ways the site’s 2025 snapshot cannot capture. Additionally, effective tariff rates on Chinese solar panels have reached **175%**,  with Southeast Asian alternatives facing rates of **34–652%** depending on country of origin. 

The site also does not appear to account for realistic deployment constraints. LBNL’s “Queued Up” 2025 Edition reports **~2,300 GW** of capacity seeking grid connection, but **only 13% of projects** that submitted interconnection requests from 2000–2019 reached commercial operations.   Median time from request to operation has grown to **5 years for projects built in 2023**.  Any modeled “optimal” portfolio must be discounted by the probability its component technologies can actually be built within relevant timeframes.

-----

## 8. How the site compares to established models — and where it genuinely excels

The site’s methodology occupies a distinct niche relative to major capacity expansion models:

|Capability           |GenX                  |PyPSA                 |ReEDS                       |US-REGEN            |The 8,760 Problem                       |
|---------------------|----------------------|----------------------|----------------------------|--------------------|----------------------------------------|
|Investment decisions |Endogenous            |Endogenous            |Endogenous                  |Endogenous          |Exogenous (enumerated)                  |
|Temporal resolution  |Up to 8,760 hrs       |Up to 8,760 hrs       |17 time-slices + 8760 module|Representative hours|Full 8,760 hrs                          |
|Transmission         |Multi-region          |Full network          |134 balancing areas         |16 regions          |None (single ISO)                       |
|Time horizon         |Single/multi-period   |Single/multi-period   |2-year steps to 2050        |5-year steps to 2050|Single year                             |
|Uncertainty coverage |~Dozens of scenarios  |~Dozens + stochastic  |~Dozens                     |~Dozens             |**5,832+ cost × 150,000+ LMP scenarios**|
|Solution transparency|Single optimum (+ MGA)|Single optimum (+ MGA)|Single optimum              |Single optimum      |**All ~33,000 mixes visible**           |

The site’s **genuine advantages** should not be dismissed. Its full 8,760-hour temporal resolution avoids the approximation errors that Bistline (2021) showed can “understate the value of broader technological portfolios, firm low-emitting technologies, wind generation, and energy storage resources.”  Its massive scenario enumeration provides uncertainty coverage orders of magnitude beyond standard practice. Its transparency allows stakeholders to explore trade-offs directly rather than trusting a solver’s single answer — addressing the critique raised by Trutnevyte et al. (2016) that cost-optimal scenarios deviated from real-world UK electricity transition by **9–23% of cumulative costs**. 

Its **critical weaknesses** relative to these models are: no endogenous capacity selection (may miss optimal combinations between enumeration grid points), no transmission (missing up to 46% cost savings from inter-regional coordination), no multi-year dynamics (cannot capture path dependencies, asset aging, or sequential investment), and no co-optimization feedback (cannot discover counterintuitive portfolio shifts that emerge when investment and operations are solved simultaneously). 

-----

## 9. Data inputs carry known biases the site should disclose

Three data quality concerns warrant explicit acknowledgment. First, if the site uses eGRID emission rates for hourly analysis, it is applying annual-average data to an hourly framework. Siler-Evans, Azevedo, and Morgan (2012) demonstrated that average emission factors **“may grossly misestimate the avoided emissions”**  — Donti, Kolter, and Azevedo (2019) found a **45% underestimate** in PJM using average rather than marginal factors.  Second, EIA Form 930 hourly generation data — the best available near-real-time source  — excludes distributed generation, has fuel-type allocation inconsistencies (sums sometimes differing from totals “by a factor of 10 or more” for smaller balancing authorities per PUDL documentation), and reports preliminary data.  Third, the site’s synthetic LMP scenarios at ISO/zonal level cannot capture intra-ISO congestion that reached **$11.6 billion across RTOs in 2022**   (Grid Strategies, 2023) or renewable curtailment that hit **3.4 million MWh in CAISO alone in 2024**  (EIA).

-----

## Conclusion: A valuable tool bounded by assumptions it should make visible

“The 8,760 Problem” makes a legitimate and underserved contribution: it democratizes access to hourly clean energy matching analysis with a transparency that no major capacity expansion model provides. Its exhaustive enumeration and massive scenario coverage offer genuine analytical value for procurement decision-makers navigating deep uncertainty. The finding that clean firm power reduces hourly matching costs is directionally consistent with the academic literature (Sepulveda et al. 2018,  Jenkins et al.).  

But the site’s conclusions are bounded by assumptions that systematically shape its results. **Independent ISO modeling overstates hourly matching costs by ignoring inter-regional coordination worth up to 46% in cost savings.**  A single weather year creates ±10% cost uncertainty   and potentially halves storage requirements versus multi-year analysis.  The clean firm advocacy rests on learning curves that nuclear power has never achieved in the West   — with every recent project overrunning by 2–4×.  The OBBBA has fundamentally altered the tax credit landscape  assumed in the 2025 snapshot,   and the GHG Protocol Scope 2 revision’s  trajectory remains contested. The site would be strengthened by explicitly bounding its findings with these limitations, presenting sensitivity analyses across weather years, disclosing its enumeration granularity, and updating to post-OBBBA economics where wind and solar credits face a hard cliff while nuclear and storage retain longer credit access.

The most constructive path forward would be to position the tool as what it genuinely is — a high-resolution screening and scenario exploration framework that complements, rather than substitutes for, capacity expansion models — while being forthright about the boundary conditions within which its findings hold.