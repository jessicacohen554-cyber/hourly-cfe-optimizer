# Claude Code Prompt Pack
## Pathway Optimizer v2 — Terminal-Candidate Refactor (P1 vs P3)

This prompt pack is designed for **Claude Code (Opus 4.7)** to refactor and extend the existing Python script **“Pathway optimizer — v2”** with deterministic, minimal-change edits while preserving output schema stability.

---

## GLOBAL CONTEXT (APPLIES TO ALL PROMPTS)

- Main file to modify:  
  **[FILE_PATHS]** (expected: `step_2_3_pathway_optimizer.py`)
- Reference optimizer:  
  **step2_2a_cost_optimization.py**
- Output schema (frozen):  
  **[SCHEMA_PATH]** (PATHWAY_OUTPUT_SCHEMA.md)
- Sample outputs for regression:  
  **[SAMPLE_OUTPUTS]** (e.g. pathway3_ep99.json.md)

**Non‑negotiables**
- Pathway 1 (P1) behavior MUST remain unchanged.
- Pathway 3 (P3) MUST be terminal‑anchored using a **membership set** (top‑K candidates), not a single fixed endpoint mix.
- All schema changes MUST be additive.
- Code edits MUST be deterministic and repeatable.

---

## Prompt 0 — Repo & Codebase Orientation (READ‑ONLY)

```text
ROLE
You are Claude Code (Opus 4.7).
Do NOT edit code in this prompt. Use planning mode only.

TASK
Scan the repository and produce a concise orientation report covering:

1. Core solver structure
   - Location and role of solve_pathway()
   - Where the per‑year argmin occurs
   - How winners[] is populated

2. Pathway differences
   - How Pathway 1 differs from Pathway 3
   - Where pathway mask logic lives (_pathway_mask)
   - Where clean‑side ratchet logic is enforced

3. Endpoint handling
   - Where endpoint_pct is defined and mapped to thresholds
   - How CFE targets are interpolated over time

4. Cost and reliability accounting
   - Where reliability_tax is computed
   - Current components included (gas capex, FOM, curtailment, etc.)
   - Where stranding_metadata is populated

5. Output & serialization
   - Where JSON output is assembled (serialize_run_result)
   - How output filenames are constructed

6. Performance architecture
   - Stage‑1 sidecar behavior (peakclean)
   - Any vectorized matrices reused in the year loop

7. Reference review
   - In step2_2a_cost_optimization.py, identify the single‑year
     “evaluate all mixes → select cheapest” logic that could be reused
     conceptually for endpoint candidate selection.

DELIVERABLE
A structured outline with file paths and function names.
No design proposals yet.
```

---

## Prompt 1 — Design Plan (NO CODE)

```text
ROLE
You are Claude Code (Opus 4.7).
Use planning mode only. Do NOT write code.

GOAL
Propose a minimal‑change design to implement **terminal‑candidate membership planning** for Pathway 3.

REQUIREMENTS
You must explicitly describe:

1. Terminal candidate generation
   - How top‑[TOP_K] endpoint candidates are computed
   - Endpoint year demand alignment (must match demand_twh_vec()[endpoint index])
   - Deterministic tie‑breaking rules

2. Candidate storage
   - Where terminal candidates live (RunConfig field, local variable, etc.)

3. Membership enforcement (P3 only)
   - How you ensure the endpoint‑year winner ∈ candidate set
   - Do NOT force a single fixed endpoint mix ex‑ante

4. Endpoint horizons
   - Support independent runs for:
     • 2040 / 90%
     • 2045 / 95%
     • 2050 / 99%
     • 2050 / 99.9%
   - No sequential layering

5. Metrics to endpoint year
   - Define cumulative cost to endpoint year
   - Define p3_relative_cost vs P1

6. Pathway 1 guardrail
   - Explicitly state what code paths remain untouched

7. Schema stability
   - List every new output field
   - Confirm all are additive and backward‑compatible

DELIVERABLE
A clear, step‑by‑step design plan with bullet points.
No code.
```

---

## Prompt 2 — Implementation (CODE CHANGES)

```text
ROLE
You are Claude Code (Opus 4.7).
You MAY write and modify code in this prompt.

OBJECTIVE
Implement the approved design with deterministic, minimal diffs.

A. CONFIG & CLI
- Add RunConfig fields:
  - endpoint_year: int (default 2050)
  - top_k_terminal_candidates: int (default [TOP_K])
- Add CLI args:
  --endpoint-year
  --top-k-terminal-candidates
  --run-matrix

B. TERMINAL CANDIDATE SELECTION (P3)
- At endpoint year:
  - Compute cost for all eligible mixes
  - Apply pathway mask
  - Select top‑K cheapest mixes
  - Use stable ordering (cost, then mix index)

C. MEMBERSHIP ENFORCEMENT (P3)
- For each terminal candidate:
  - Run a forward solve with endpoint winner forced
  - Compute cumulative cost to endpoint year
- Select the candidate with lowest cumulative cost
- Re‑run once to produce final outputs

D. PATHWAY 1
- Do NOT alter argmin logic or constraints
- Only additive outputs allowed

E. OUTPUT ADDITIONS (ADDITIVE ONLY)
1. Ratepayer metric
   - annual_usd_per_mwh = net_annual_cost_usd / (demand_twh * 1e6)
   - Add per year + headline series

2. Reliability tax symmetry
   - Add: reliability_tax['clean_firm_reliability_cost_usd']
   - Computed from clean‑firm tranche vintages

3. Stranding symmetry (P1 only)
   - Add VRE regret metric to stranding_metadata
   - Use locked‑vs‑current LCOE proxy (document clearly)

F. RUN MATRIX OUTPUT
- For each ISO, produce:
  - 8 runs (4 endpoints × P1/P3)
  - One comparison JSON with:
    • cumulative costs to endpoint
    • p3_relative_cost
    • annual $/MWh series side‑by‑side

G. FILE NAMING
- Per run:
  [ISO]__pathway{p}__ep{ENDPOINT_PCT}__y{ENDPOINT_YEAR}.json
- Comparison:
  [ISO]__comparison__matrix.json

DELIVERABLE
- Code changes only
- Brief summary of modified functions
```

---

## Prompt 3 — Testing & Regression

```text
ROLE
You are Claude Code (Opus 4.7).
You MAY add tests.

TEST REQUIREMENTS

1. Determinism tests
- Terminal candidate selection stable across runs
- Tie‑breaking reproducible

2. Membership tests
- P3 endpoint winner ∈ candidate set
- Forced endpoint respected

3. Regression tests
- Pathway 1 outputs unchanged except additive fields

4. Metric correctness
- annual_usd_per_mwh arithmetic exact

5. Schema checks
- JSON serializable
- No existing keys removed or renamed

6. Matrix sanity
- 4×2 runs present
- p3_relative_cost computed to endpoint year

DELIVERABLE
Test plan + test files (or clear pseudo‑tests if framework absent)
```

---

## Prompt 4 — Performance Guardrails

```text
ROLE
You are Claude Code (Opus 4.7).

CONSTRAINTS
- Do NOT break Stage‑1 sidecar behavior
- No per‑mix‑per‑year Python loops
- Terminal candidate logic must be O(n_mixes log n_mixes) once per run

DOCUMENT
- Big‑O impact
- Any new allocations
- Why chosen approach is lowest risk
```

---

## Prompt 5 — Documentation / Change Log

```text
ROLE
You are Claude Code (Opus 4.7).

DOCUMENT
1. New config fields & CLI args
2. Terminal‑candidate membership logic (plain English)
3. Why annual $/MWh is the headline metric
4. Reliability tax symmetry change
5. VRE stranding metric for P1
6. Run‑matrix output structure

DELIVERABLE
Concise developer‑facing changelog + inline comments
```

---

# END PROMPT PACK
