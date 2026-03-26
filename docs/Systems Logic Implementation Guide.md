# Systems Logic Implementation Guide: Noesis to Dianoia

This guide facilitates the transition from **Noesis** (holistic synthesis) to **Dianoia** (linear proof). Use the following prompts to generate repo-ready components that adhere to the **Systems Architect** aesthetic: 2px saturated outlines, 12% alpha fills, and strictly sans-serif typography.

## 1. Global Visual DNA (System Constraints)

**Typography:** Plus Jakarta Sans (Headings), DM Sans (Body).

**Linework:** 2px solid outlines (e.g., --solar, --red, --cyan).

**Fills:** 12% opacity (rgba(..., 0.12)) matching the saturated outline color.

**Physics:** Use stroke-dashoffset for flow vectors and pulse for FOAK signals.

## 2. Integrated Narrative & Implementation Prompts

| Slide | System Logic & Script | Visual Archetype | Claude Implementation Prompt (Copy/Paste) |
| --- | --- | --- | --- |
| 1 | The Grid is Not a Bank. "The grid is a live, interconnected machine where carbon intensity varies wildly by the hour and location. Most clean energy policies ignore this physics, treating all MWh as equal." | Spatio-Temporal Load Map. (24h x Nodes heatmap). | "Build a React heatmap for a 24h x 5 Node grid. Script: Include the 'Grid is Not a Bank' narrative in a left-hand panel. Aesthetic: 2px saturated outlines, 12% fills. Logic: Toggle 'Illusion' (flat 350g/kWh) vs. 'Reality' (volatile nodal data). Use a CSS 'scanline' animation. Tooltip shows 'Nodal Intensity' in --font-data." |
| 2 | The '100% Clean' Illusion. "Annual accounting matches abundant daytime solar with nighttime demand. The math works on paper, but physically, the company relies entirely on the dirty grid at night." | Diurnal Matching Chart. Hourly VRE vs. Load. | "Create an SVG Area Chart. Script: Include the '100% Clean Illusion' narrative. Layers: 2px dashed line for 'Annual Avg', solid 2px area for hourly shape. Shading: Fill the gap between 'Real Time' and 'Average' with --red (0.12 alpha) for night and --solar (0.12 alpha) for day. Add an overlay graphic: '100% Annually Matched'." |
| 3 | The VRE-Gas Trap. "By only deploying VRE, we crash LMPs during easy hours, forcing the grid to keep redundant gas online for the dark hours. This capital is now 'locked in'." | Active Circuit Loop. Causal flow with feedback. | "Build an interactive SVG Circuit. Script: Include the 'VRE-Gas Trap' narrative. Nodes: VRE, Market, Gas. Paths: 2px 'marching ants'. Interaction: Clicking 'VRE' triggers a feedback loop returning from 'Gas' to 'VRE', turning the circuit --red. Icons: Wind, Solar, Flame. Label the outcome: 'Structural Reliance'." |
| 4 | The Stranded Asset Cliff. "In regions like PJM, you might successfully deplete the coal wall with cheap wind. But you are left with a massive gas fleet and redundant VRE that eventually become stranded." | MAC Curve Waterfall. Abatement steps vs. targets. | "Generate a horizontal waterfall chart. Script: Include the 'Stranded Asset Cliff' narrative. Steps: 'Wind' (Green), 'Solar' (Amber). The Cliff: A 2px red dashed vertical line at 80% mark. Visual: Beyond the cliff, bars turn into --fossil-gas with 'Stranded' warning icons. Use --font-data for the 95%+ target label." |
| 5 | The Integrated Portfolio. "The best path is incentive structures grounded in spatial and temporal realities. We must pay a necessary clean premium for FOAK Firm Clean tech today." | FOAK Signal Matrix. 2D Plot (Cost vs. Resiliency). | "Create a 2D Scatter Plot. Script: Include the 'Integrated Portfolio' narrative. X-Axis: Resiliency. Y-Axis: Cost. Plot 'VRE-only' vs. 'Integrated Portfolio'. Animation: The 'Firm Clean' node must have a 2px saturated border and a pulse-glow to represent the FOAK technology signal. Use a light grid background." |
| 6 | The True Endpoint. "A high-fidelity grid where corporate procurement signals have solved the temporal and spatial dependency. We minimize waste and achieve physical decarbonization." | The Flush-Fit Integral. High-fidelity Stacked Area. | "Generate a high-fidelity Stacked Area chart. Script: Include the 'True Endpoint' narrative. Goal: Stack Wind, Solar, Storage, and Firm Clean so the total sum is perfectly 'flush' against a flat Load line. Visual: 2px saturated borders between segments and 12% alpha fills. Badge the result: 'System Integral Solved' in .stat-value style." |

## 3. Data Integration Strategy

When prompting Claude for these components, ensure you specify:

**Shared CSS Reference:** "Inherit variables from the site's shared stylesheet."

**Data Source:** "Use the specific regional pathway data from regional_pathways.json (or your specific analysis)."

**State Management:** "Use React useState to toggle between the 'Illusion' and 'Reality' states to visualize the systems gap."
