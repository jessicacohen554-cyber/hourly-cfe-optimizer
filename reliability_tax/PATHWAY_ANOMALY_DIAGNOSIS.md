# Step 2.3 — Pathway-anomaly diagnosis

**Verdict: B (expected v2 behavior, no bug).**

Session D's observation — that all 5 pathways produce identical
`fleet_size_mw` / `achieved_cfe_pct` / `reliability_tax` / `endpoint_mix` /
`stranded_capex` in the ERCOT sweep — is real but **ISO-specific**. The
pathway constraint is correctly applied in the argmin; it simply does not
bind in ERCOT (and SPP) at most endpoints because the unconstrained
cost-optimal mix already satisfies P1's no-new-clean-firm rule. In NEISO,
PJM, MISO, and at select SPP endpoints, the same code produces clearly
divergent mixes between P1/P1a and P2a/P2b/P3. That divergence is
mechanical proof the mask is being honored.

CAISO is excluded from this diagnosis — only 8 of 50 pathway JSONs had
landed when this ran. The six ISOs with complete v2 sweeps (ERCOT, MISO,
NEISO, NYISO, PJM, SPP) are sufficient to settle the question.

## Where the constraint is applied

`scripts/step_2_3_pathway_optimizer.py`:

| Line(s) | Code | Role |
|---|---|---|
| 723–724 | `_pathway_firm_floor_pct(iso)` returns `pc.GRID_MIX_SHARES[iso]['clean_firm']` | Existing-nuclear floor (grandfathered) |
| 727–752 | `_pathway_mask(ef, cfg)` | Builds `(N_YEARS, n_mixes)` bool mask |
| 732 | `floor = _pathway_firm_floor_pct(cfg.iso) + 0.01` | 1-pp buffer above existing share |
| 733–734 | `no_clean_firm = cf <= floor`; `no_ccs = ccs <= 0.5` | Per-mix eligibility |
| 736–738 | P1 / P1a: `mask[:] = no_clean_firm & no_ccs` (all years) | Hard lock — no new firm clean, ever |
| 739–751 | P2a / P2b / P3: locked like P1 until `NOAK_YEAR[pw] - 5`, then `ones` | Time-gated unlock |
| 815 | `pmask = _pathway_mask(ef, cfg)` | Applied in `solve_pathway` |
| 832 | `valid = pmask & cfe_mask` | AND with per-year CFE target mask |
| 833 | `cost_matrix = np.where(valid, cost_matrix, np.inf)` | Ineligible rows → +∞ cost |
| 836–844 | `winners[yi] = int(np.argmin(cost_matrix[yi]))` | argmin over masked cost |
| 839 (relax) | Relaxed-CFE branch still preserves `pmask[yi]` | Pathway lock cannot be relaxed away |

The mask feeds straight into the argmin via the `+∞` sentinel — any row
outside the pathway envelope is unreachable by `np.argmin`. This is the
correct way to bind a linear feasibility constraint onto a grid-search
optimizer.

## Fingerprint table — ERCOT, PJM, NEISO, MISO, NYISO, SPP

Fingerprint format is `[cf | solar | wind | hydro | offshore_wind | geo | ccs | sol_b4 | sol_b8 | wind_b4 | wind_b8] / [batt4 | batt8 | LDES | H2]` (percentages of 2050 mix; storage is dispatch share).

| iso   | pathway | endpoint | fleet (MW) | achieved CFE | new clean_firm built (TWh/yr) | selected_mix_fingerprint |
|-------|:-------:|:--------:|-----------:|-------------:|------------------------------:|---|
| ERCOT | P1 | ep90 | 139,983 | 90.07% | 0.0 | `[0.0\|20.0\|80.0\|0\|0\|0\|0\|0\|0\|0\|0] / [0\|0.15\|0.30\|0]` |
| ERCOT | P3 | ep90 | 139,983 | 90.07% | 0.0 | `[0.0\|20.0\|80.0\|0\|0\|0\|0\|0\|0\|0\|0] / [0\|0.15\|0.30\|0]` |
| PJM   | P1 | ep90 | 131,599 | 91.00% | **0.0** | `[29\|30\|20\|2\|0\|0\|0\|0\|1\|9\|29] / [0\|0\|0.20\|0]` |
| PJM   | P3 | ep90 |  94,104 | 90.03% | **489.6** | `[79\|0\|10\|2\|3\|0\|0\|1\|1\|0\|0] / [0\|0\|0\|0]` |
| NEISO | P1 | ep90 |  15,111 | 90.04% | 0.0 | `[21\|0\|11\|5\|31\|0\|0\|19\|9\|9\|9] / [0\|0\|0\|0]` |
| NEISO | P3 | ep90 |  10,114 | 90.04% | 0.0 | `[24\|27\|20\|4\|25\|0\|0\|0\|0\|0\|0] / [0.04\|0.03\|0.70\|0]` |
| MISO  | P1 | ep99 |  66,630 | 99.00% | 0.0 | `[9\|1\|109\|0\|0\|0\|0\|0\|9\|29\|9] / [0\|0.15\|1.00\|0]` |
| MISO  | P3 | ep99 |  64,550 | 99.01% | **45.6** | `[19\|10\|30\|2\|0\|0\|0\|19\|1\|39\|9] / [0.10\|0.15\|1.00\|0]` |
| NYISO | P1 | ep90 |  19,742 | 90.00% | 0.0 | `[18\|30\|7\|16\|25\|0\|0\|0\|0\|0\|0] / [0.07\|0.15\|1.00\|0]` |
| NYISO | P3 | ep90 |  19,742 | 90.00% | 0.0 | `[18\|30\|7\|16\|25\|0\|0\|0\|0\|0\|0] / [0.07\|0.15\|1.00\|0]` |
| SPP   | P1 | ep90 |  25,229 | 91.57% | 0.0 | `[0\|60\|40\|6\|0\|0\|0\|0\|0\|0\|0] / [0\|0.15\|0\|0]` |
| SPP   | P3 | ep90 |  25,229 | 91.57% | 0.0 | `[0\|60\|40\|6\|0\|0\|0\|0\|0\|0\|0] / [0\|0.15\|0\|0]` |

## Where the constraint clearly binds

Per-endpoint unique-fingerprint counts across `{P1, P1a, P2a, P2b, P3}`, for
the six ISOs with full sweeps:

| ISO   | existing cf share | floor (cf +1pp) | endpoints w/ ≥2 unique FPs | bind evidence |
|-------|---:|---:|---|---|
| ERCOT |  8.6% |  9.6% | ep60, ep70 (P3 reaches 9% cf while P1 stays at 0%) | weak — only at low CFE |
| PJM   | 32.1% | 33.1% | ep90, ep99 (P3 builds **489.6 TWh/yr** of new nuclear at ep90; P1 zero) | **strong** |
| NEISO | 23.8% | 24.8% | every endpoint (ep60–ep99p9, all 10) | **strong** |
| MISO  | 13.1% | 14.1% | ep95, ep97p5, ep99, ep99p9 (P3 builds up to 21% cf; P1 capped at 9%) | **strong** |
| NYISO | 18.4% | 19.4% | none | bind never tightens — existing nuclear already sits at cost-optimum |
| SPP   |  5.2% |  6.2% | ep97p5 only | weak — SPP's wind (CF ≈ 0.42) dominates |

PJM ep90 is the smoking gun. Same code, same ISO, same endpoint —
**P3 adds 489.6 TWh/yr of new clean_firm generation, P1 adds zero**.
The only thing that changed is the pathway label, which flows through
`_pathway_mask` into `np.where(valid, cost_matrix, np.inf)`. If the mask
were being ignored, P1 and P3 would return the same mix. They don't.

## Economic interpretation — why ERCOT reads as "identical"

ERCOT's cost-optimal 2050 mix at ep90 is 20% solar + 80% wind + 15%
battery8 dispatch + 30% LDES dispatch, with **zero new clean_firm**
above the 9.6% existing-nuclear floor. Three reinforcing reasons:

1. **ERCOT has the best onshore wind in the country** — annual CF ≈ 0.38
   per `pipeline_config.py:927`. Every marginal MWh of wind arrives
   cheaper than every marginal MWh of new nuclear (IRA PTC-inclusive
   LCOE ≈ $40/MWh for ERCOT wind vs. $80–100/MWh for new NOAK nuclear).
2. **LDES (10-hour) plus 8-hour battery hybrids** close the residual
   gap efficiently at 90% CFE — you don't need a dispatchable firm
   resource to clear the worst-hour residual, because storage already
   does it for a fraction of the capex.
3. **ERCOT's existing nuclear is already below the floor (8.6% < 9.6%)**.
   P1 doesn't force retirement; it just prohibits *new* firm construction.
   Since the unconstrained optimum also doesn't want new firm, P1 and P3
   converge on the same 2050 solution.

The pattern breaks in PJM because PJM's wind (CF ≈ 0.30) and solar
(CF ≈ 0.17) are weaker, and PJM's existing nuclear fleet is large
(32.1%) — the unconstrained optimizer wants to *double down* on
nuclear, which P3 permits (→ 79% cf, 489.6 TWh/yr of new build) but P1
forbids (→ stays at 29% cf, all existing-fleet, compensates with
225 GW of wind+battery8 hybrids).

## Why Session D's ERCOT-only sanity saw "identical everywhere"

The ep60/ep70 ERCOT signatures actually differ between P1 and P3
(P1 picks 0% cf, P3 picks 9% cf). Session D's SESSION_D_PROGRESS.md:55
claim of "all 5 pathways at every endpoint" is slightly overstated —
it's true for ep75–ep99p9 but not ep60/ep70. The same is true for SPP
at ep97p5. These small cracks in the "identical everywhere" claim are
already consistent with verdict B: the mask bites only when the
constraint actually differs from the unconstrained optimum.

## Action items

None for this session. The code is correct as written.

Two adjacent things worth noting for whoever picks up Session C-style
work (not fixes, just documentation upgrades):

- **Dashboard copy** should not imply that ERCOT's "5 pathways → same
  answer" pattern is a universal v2 finding. It's an ERCOT artifact.
  NEISO, PJM, and MISO tell different stories and should anchor any
  pathway-compare section.
- **The `_pathway_firm_floor_pct` + 1pp buffer** at line 732 deserves a
  one-line comment explaining why (to allow EF rows that sit at exactly
  the grid-mix share, which otherwise would be filtered out by floating
  point). Not urgent.

## Addendum — Phase-1 probe (2026-04-21)

A follow-up session re-opened the question under the hypothesis that the
collapse might happen *downstream* of the argmin (i.e. optimizer picks
different mixes per pathway, but a serializer helper flattens them). The
hypothesis was falsified by a 2-combo probe: temporary prints added to
`solve_pathway` around the `winners` array, run with `ISO_FILTER=ERCOT`
for P1 and P3 at ep90:

```
ERCOT P1 ep90:  winners_sum=3219398  end_w=123823  fleet=139982.81 MW
                winners[0..5]=[123823,123823,123823,123823,123823,123823]
                archetype_key=2312f832ee754584
ERCOT P3 ep90:  winners_sum=3219398  end_w=123823  fleet=139982.81 MW
                winners[0..5]=[123823,123823,123823,123823,123823,123823]
                archetype_key=2312f832ee754584
```

Both pathways argmin to **the same EF row (123823) every year**
(`winners_sum` is identical to the byte). Row 123823 is the
`[0% clean_firm | 20% solar | 80% wind]` mix — P1-compliant, P3-optimal
unconstrained. The archetype lookup, gas-sizing pass, ledger build, and
JSON writer all receive identical inputs, so they emit identical outputs
by construction. There is no downstream flattening.

**Conclusion reaffirmed:** verdict B. The ERCOT "5 pathways → same
headline" pattern originates at the argmin, not downstream of it. No
code fix applies. The landed JSONs for ERCOT/SPP/NYISO at endpoints
where the pathway mask doesn't bind are correct.
