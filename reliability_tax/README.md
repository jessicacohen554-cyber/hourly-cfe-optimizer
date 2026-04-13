# The Reliability Tax

A sub-project inside `hourly-cfe-optimizer` quantifying the cost gradient of pushing corporate/grid CFE (Carbon-Free Energy) matching from ~90% up toward 99.9% across 2025–2050 and seven US ISOs. The central question: **what is the marginal and total cost of each extra point of CFE beyond 90%, and when does a VRE+batteries-only strategy stop being the cheapest answer?**

This README is the authoritative entry point for the sub-project. Methodology details, pathway implementation, and stranding math will be locked in subsequent prompts (1B+) and mirrored into the main `SPEC.md`.

---

## Locked invariants

The following decisions are load-bearing and apply to every script, figure, and writeup in this sub-project. They must not drift.

1. **Endpoint targets** — CFE ≥ 90%, 95%, 97.5%, 99%, 99.9% by 2050.
2. **Planning horizon** — 2025–2050 (25 years; 26 year-indices including the 2025 baseline).
3. **Pathways under comparison**:
   - (1) **VRE + batteries only** — no new clean firm at any point.
   - (2a) **Behavioral pivot** — VRE-first until the 90% CFE plateau, then pivot to clean firm.
   - (2b) **Economic pivot** — VRE-first until marginal `$/CFE%` exceeds the prevailing clean firm LCOE, then pivot.
   - (3) **Clean firm proactive** — clean firm deployed from year 1 alongside VRE.
4. **Clean firm bucket** — Nuclear + CCGT+CCS + geothermal, subject to existing regional constraints (CCS caps from `pipeline_config.CCS_CAP_TWH`; geothermal is CAISO-only). **Offshore wind is NOT in the clean firm bucket** — it is VRE, available to all pathways.
5. **Cost basis** — Real 2025 USD. No inflation adjustment.
6. **Cost reporting** — Undiscounted cumulative 2025–2050 **plus** NPV at 5%, 7%, and 9% real. **Objective = NPV@7%.**
7. **ISO scope** — Fully ISO-parametric. Smoke-test on a single ISO, then run all seven (`CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP`).
8. **Stranding scope & thresholds**:
   - Stranded fossil = **new-build gas only** with capacity factor <20% in 2050. Existing fleet is out of scope.
   - VRE stranding = curtailment >30% in 2050.
   - Transmission stranding = underutilized new-build transmission (precise definition locked in 1B).
9. **Demand growth sensitivity** — Used in Section 2 only, reusing the existing L/M/H values from `pipeline_config.DEMAND_GROWTH_RATES`. No new growth rates are invented.

---

## Reusable infrastructure map

The Reliability Tax analysis layers on top of the existing 8-step pipeline. Nothing in the main pipeline should need to change to support it. Key reusable pieces:

| File / function | Role | How Reliability Tax uses it |
|---|---|---|
| `scripts/pipeline_config.py` | Single source of truth for cost tables, caps, demand growth, thresholds, SBTi year mapping | Import constants directly; never redefine LCOEs, caps, or growth rates locally |
| `scripts/procurement_utils.py` → `build_25yr_trajectory(iso, strategy_fn, ...)` | Year-by-year 2025–2050 iterator with demand growth, SBTi target mapping, Wright's Law integration, nuclear subsidy roll-off | Backbone of pathway simulation — each of the four pathways is a `strategy_fn` plugged into this iterator |
| `scripts/pipeline_config.py::THRESHOLD_TARGET_YEARS` | SBTi milestone ladder (2030→50%, 2035→70%, 2040→90%, 2045→95%, 2050→99.9%) | Defines the target schedule; pathway (3) deviates by front-loading clean firm, pathways (2a/2b) branch after the 90% milestone |
| `scripts/pipeline_config.py::DEMAND_GROWTH_RATES` | L/M/H annual growth per ISO | Section 2 sensitivity sweeps; otherwise Medium is the default |
| `scripts/pipeline_config.py::CCS_CAP_TWH` + `GEOTHERMAL_ISOS` + `OFFSHORE_ISOS` | Regional technology caps | Enforced on the clean firm bucket (CCS, geothermal) and on VRE (offshore wind is VRE, not clean firm) |
| `data/step2.1-ef/step_2_1_EF_{ISO}_{THRESHOLD}.parquet` | Efficient frontier mixes per ISO × threshold | Starting point for VRE-only pathway mix selection and for constructing the 90% → 99.9% climb |
| `data/step2.2-cost/step_2_2a_DG_{ISO}_{THRESHOLD}.parquet` | Cross-eval of EF mixes under 5,832 cost scenarios | Source for per-mix, per-scenario cost curves used to compute marginal `$/CFE%` for the economic-pivot trigger |
| `data/step3-dispatch/{ISO}_dispatch_cache.parquet` | Hourly dispatch + per-resource matched/surplus/charge profiles | Feeds curtailment calculations for the ">30% curtailment" VRE stranding threshold |
| `scripts/step5_2e_wrights_law_curves.py` + `data/step5-wrights/wrights_law_curves.parquet` | Per-technology learning curves | Used by the 25-year iterator to decline VRE and storage capex over time |

### Not yet modeled (scope of later prompts)

- Proactive clean-firm deployment from year 1 (existing work only places clean firm reactively to SBTi thresholds).
- Pivot-trigger logic — behavioral (at 90% plateau) and economic (marginal `$/CFE%` > clean firm LCOE).
- New-build-only gas capacity factor stranding tracker (existing fleet is out of scope).
- VRE curtailment-level stranding tracker at the 30% threshold.
- Transmission stranding bookkeeping for new-build lines.
- Cross-pathway NPV@5/7/9% comparison and cost-gradient visualization.
- Reliability Tax dashboard page.

---

## Directory structure (planned)

```
reliability_tax/
├── README.md          # this file
├── scripts/           # empty for now; pathway sim + cost/stranding modules land here in 1B+
├── data/              # empty for now; sub-project outputs (not raw inputs)
├── notebooks/         # empty for now; exploratory analysis
└── results/           # empty for now; final NPV tables + figures
```

Nothing except `README.md` exists yet. Subdirectories will be created lazily as prompts 1B+ need them. Raw input data continues to live in the main `data/` tree — Reliability Tax only writes to `reliability_tax/data/` and `reliability_tax/results/`.

---

## Out of scope for Prompt 1A

Prompt 1A is strictly discovery + documentation. The following are explicitly deferred:

- No pipeline scripts, notebooks, or data generation.
- No pathway simulation code or pivot-trigger math.
- No stranding calculations.
- No dashboard page or visualizations.
- No changes to the main repo `README.md`, the existing pipeline steps, or any dashboard HTML.
- No PR creation.

---

## Next steps

Prompt **1B** (to be issued by the user) will lock the methodology: pathway implementation, pivot-trigger math, stranding calculations, and the smoke-test ISO. Until 1B is locked, no code is written and no runs are launched.

Per the repo's documentation-first workflow, every design decision made in 1B+ must be mirrored back into this README and into the Reliability Tax section of `SPEC.md` immediately upon confirmation.
