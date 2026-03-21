##USER REVIEW & EXPECTATIONS 
#Issue -- fleet_scenario.html page is not functioning correctly… 

#Context / new changes: 
    - I have uploaded a new market-simulator/data/CEG_fleet_rosetta.csv which includes constellation owned capacity by plant. Plant naming convention may make cross referencing renewable assets difficult but that is OK. Fossil assets, especially with CAMPD data and facility IDs. This should serve as the primary source of truth on the CEG fleet. Capacity attributed to constellation in this dataset is based on equity share ownership, which is how emissions should be derived based on campd data and then for other fossil plants, derived from EIA 923 data x generic EPA emission rates for each fuel if 923 does not provide CO2 data
    - I have run a new set of 1215 sweep data available here market-simulator/data/results that will now include annual data instead of 5 year increments. It should be leveraged for this. this shoild be avaiable to read from when we are ready to execute prolpts. 
#Deliverable: ultimately I want this integrated into the market-simulator UI seamlessly but in the interim to test and troubleshoot, I want out to use the data / scripts / etc. to produce a publishable version of the fleet_scenario.html file available via the main dashboard site link, but NOT linked in Nav or searchable. 

#The vision: use market trajectory 1215 sweep data to understand probabilistic dispatch and generation stars for a baseline, p50 with confidence intervals case to understand emissions trajectory of the fleet without any intervention. Then enable the user to layer in decarbonization scenarios on top of that using CCs retrofitting on CCGT assets based on actual capacity, capacity derate, capacity factor, and carbon capture assumptions. Captured carbon is treated as reduced emissions. Retired plants would immediately drop to 0 emissions in the fleet scenario even if they were still economic under some scenarios of the market sweep. The vision is that the dashboard  shows what’s achievable with intervention on top of a baseline market trajectory. 

#Section 1 
- draft understanding of emissions calculations. First: i need to you to give me a step by step understanding of how you are calculating emissions from the constellation fleet in market-simulator/frontend/fleet-scenarios.html and based on my questions below, deduce ways to resolve the inconsistencies and series issues occurring with the dashboard. 

    1. What scripts and datasets does it rely on for upstream values? 
        1.1 Note the actual baseline case that displays on the fleet scenario UI should always be available to select on the fan chart and should be built purely on market trajectory 1215 sweep. 
    2. how are constellation fleet emissions derived from the market sweep results?
        2.1 is it plant specific dispatch And if so which data sources are you using to cross-reference constellation plants to dispatch data?
    3. what input data files are being used to identify and query the constellation fleet for the sidebar on the page? 
        3.1 are those aligned precisely to the same data files being used per my question in 2 re: derivation of fleet emissions from market sweep results?  
        3.2 Where do the capacity mw in the sidebar come from and why are so many 800? 
    4. How are you calculating emissions based on CCS retrofit and what variables are you using for derate, capture rate, and capacity factor? 
        4.1 right now I’m seeing huge emissions increases when I install CCS which doesn’t make sense… you’re not actually applying the capture as a reduction. 
        4.2 we should add a panel on top of fleet configuration sidebar that allows us to set capacity derate %, capacity factor % and carbon capture % for all CCS retrofit selections, which can default, reset in different scenario runs. 

#section 2
- Draft a plan to align to context /new changes and end-goal vision in terms of code / script changes, frontend uI updates, ensuring single source of truth across ALL scripts for CEG fleet assets and owner capacity: simulator/data/CEG_fleet_rosetta.csv. See above for other major edits. 

#section 3
- Draft a series of implementable prompts to achieve vision that can be leveraged in individual sessions to get to an intermediate state of dashboard/fleet_scenarios.html for live beta testing and edits 

##CLAUDE RESPONSE
#TASKS
    1. reiterare understanding of vision and interim state need to publish fleet_scenario.html via dashboard but not on nav or searchable for troubleshooting 
    2. Section 1
        respond to all questions 
    3. Section 2
        draft plan 
    3. Section 2
        draft prompts 


