## Task: Step 2.1d — Low-CF Augmentation for Nuclear-Heavy ISOs

### Read first
- CLAUDE.md → SPEC.md → LESSONS.md
- scripts/step2_1b_augment_thin_ef.py (pattern to follow — perturbation + interpolation)
- scripts/step1_pfs_generator.py lines 1-80 (scoring functions: batch_hourly_scores, _score_with_all_storage)
- scripts/pipeline_config.py (GRID_MIX_SHARES, UPRATE_CAP_TWH)

### Problem
The Step 1 mix generator used CF_WINDOW floors that excluded low-nuclear
mixes for three ISOs:

    NEISO: cf floor = 20%  →  P1 stalls at 89.7% CFE (9.3pp gap)
    NYISO: cf floor = 10%  →  P1 stalls at 96.6% CFE (2.4pp gap)
    PJM:   cf floor = 30%  →  P1 stalls at 74.9% CFE (24.1pp gap)

ISOs with cf starting at 0 (CAISO, ERCOT, SPP, MISO) reach 99% fine.
The pool has zero mixes below cf=19 for NEISO, cf=9 for NYISO, cf=29
for PJM. The P1 pathway mask tightens each year as fixed nuclear TWh
becomes a smaller share of growing demand. Once the mask drops below
the pool's minimum cf level, zero candidates remain and CFE freezes.

Additionally, P1 stalls partly because the greedy argmin locks in a
composition (e.g. wind-heavy) early on, and the ratchet prevents
switching to the different shape (solar/storage-heavy) that low-cf
mixes use at 90%+ scores. At NEISO band 90: cf=19 mixes have median
wind=19%, but the locked winner has wind=32%. Only 8.4% of cf=19
mixes have wind≥32. The ratchet makes the composition transition
impossible.

### Approach — perturbation, not full grid search
Write scripts/step2_1d_lowcf_augment.py following the step2_1b pattern:

1. SEED SELECTION: For each target ISO × each EF band ≥ 50%, load
   existing mixes at the current cf floor (cf=20 for NEISO, cf=10 for
   NYISO, cf=30 for PJM). Also load mixes 1pp above (cf=21, cf=11,
   cf=31) as additional seeds.

2. CF REDUCTION PERTURBATION: For each seed, generate variants with
   cf reduced by 1–5pp (clamped at 0). For each cf reduction of Δ,
   redistribute Δ across the non-CF resources using three strategies:
     a. Proportional: scale up all existing non-CF resources by Δ/total
     b. Solar-heavy: add Δ to solar (or solar_batt4/solar_batt8)
     c. Wind-heavy: add Δ to wind (or wind_batt4/wind_batt8)
     d. Storage-heavy: add Δ to solar_batt4 + wind_batt4 equally
   This gives ~12-20 variants per seed per Δ step — not millions.

3. INTERPOLATION: Pairwise 25/50/75% blends between the new low-cf
   variants (same logic as step2_1b generate_interpolations).

4. SCORING: Use batch_hourly_scores for resource-only, then
   _score_with_all_storage for variants with storage dispatch > 0.
   Same chunked pattern as step2_1b score_with_storage.

5. DEDUPLICATION: int16 resources + 20x-scaled int32 storage, same
   as step2_1b deduplicate().

6. OUTPUT: Write to step_2_1_EF_{ISO}_{band}_interp_lowcf.parquet
   — the pool loader in step2_3 already picks up *_interp_*.parquet
   files and builds peakclean sidecars for them automatically.

7. After writing: run step2_3a_regenerate_peakclean.py --iso {ISO}
   for each target ISO so the peakclean sidecars cover the new mixes.

### Key constraints
- Keep total new mixes per ISO under ~500K (not millions).
- Existing seeds × 5 cf steps × ~15 redistribution variants × 3
  interpolation blends ≈ a few hundred thousand per ISO per band.
  Profile one band before running all.
- Don't modify step1_pfs_generator.py or its CF_WINDOW — this is
  additive augmentation only.
- Output schema must exactly match existing EF parquets: int16 for
  resource cols, float64 for storage dispatch and hourly_match_score,
  string for iso and pareto_type.

### Validation
After running, verify:
  1. NEISO band 90 has cf levels [0..20] (was [19,20] only)
  2. PJM band 90 has cf levels starting below 29
  3. Low-cf mixes at band 90+ actually exist and have diverse resource
     compositions (not all identical shapes)
  4. Rerun NEISO P1 ep99 — CFE should exceed 89.7%

### Also: beam search prompt for Step 2.3

After the augmenter lands and is verified, the Step 2.3 solver needs
a beam search to avoid the composition-lock problem. The current
greedy argmin picks one winner per year; the ratchet locks that shape
permanently. A beam search carries the top K candidates (K≈50) per
year, each with independent ratchet floor state, and picks the lowest
total-cost trajectory at the endpoint. This is a SEPARATE task —
don't implement it in this session. Just note it as the next step
after the pool augmentation is verified.

Design notes for the beam search (deferred):
- Score the pool ONCE per year (beam-independent). Apply per-beam
  ratchet masks as numpy broadcasts post-scoring — not K separate
  kernel calls.
- beam_width=1 must reproduce current greedy results exactly.
- BeamEntry = (mix_idx, cum_cost, floor_state, history[]).
- Foresight solver unchanged — beam is myopic-path only.
