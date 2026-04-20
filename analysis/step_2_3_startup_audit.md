# Step 2.3 Startup Hang — Audit Memo
**Date:** 2026-04-20  
**Scope:** Read-only code trace, `python3 scripts/_sweep_pathways_inproc.py` entry → first `[solve]` line  
**Method:** Static analysis of every call in the startup path; no scripts executed  

---

## Summary of Findings

Six blocking call sites exist between process start and first `[solve]`. They fall into three classes:

| # | Class | Worst-case silence | Timeout protection? |
|---|-------|-------------------|---------------------|
| 1 | Dispatch-cache parquet read | 30–120 s | None |
| 2 | EIA profile deserialization | 5–30 s each (×3) | None |
| 3 | Numba JIT — dispatch kernels (first run) | 30–120 s | None |
| 4 | Numba JIT — cost kernels (first solve) | 40–180 s | None |
| 5 | EF parquet reads via `_call_with_timeout` | up to 30 s (×N thresholds) | 30 s per file |
| 6 | `_TIMEOUT_EXECUTOR` starvation (sweep phase) | up to 60 s per queued call | 60 s per call |

Items 1–4 all print nothing while blocked. The first `[solve]` line does not print until `run_pathway` begins the year loop — after all of 1–5 complete and item 6's first JIT slot opens.

---

## Traced Call Path (annotated)

```
python3 scripts/_sweep_pathways_inproc.py --iso ERCOT
  │
  ├─ MODULE IMPORT: step_2_3_pathway_optimizer
  │    line 250–252  _TIMEOUT_EXECUTOR = ThreadPoolExecutor(max_workers=2)
  │                  ← spawns 2 OS threads; fast, not a hang
  │
  ├─ MODULE IMPORT: step2_2a_cost_optimization
  │    lines 1803–2217  @njit(parallel=True, cache=True) function BODIES defined
  │                     ← decorator registration only; JIT deferred to first call
  │
  ├─ main() → sweep_one_iso() → prewarm_caches(iso, combos)   [_sweep_pathways_inproc.py:103]
  │    prints: "[sweep] {iso}: prewarming caches"
  │
  │    prewarm_caches(iso, combos)   [step2_2a_cost_optimization.py:305]
  │      prints: "[prewarm] {iso}: entering prewarm ..."
  │
  │      ► BLOCKING CALL 1 ◄  _ensure_common_data(iso)   [step2_2a_cost_optimization.py:332]
  │         │  → _get_worst_hour_state(iso)   [step2_2a_cost_optimization.py:211]
  │         │      → _du.load_dispatch_cache(iso, require_version=_du.CACHE_VERSION)
  │         │          [step2_2a_cost_optimization.py:185]
  │         │          → dispatch_utils.load_dispatch_cache()   [dispatch_utils.py:1136]
  │         │              → pq.read_table(path)   [dispatch_utils.py:1149]
  │         │                  then: Python loop over all keys/columns   [lines 1156–1172]
  │         │
  │         └─ if demand_data is None:   [step2_2a_cost_optimization.py:212]
  │               → dispatch_utils.load_common_data()   [dispatch_utils.py:87]
  │
  │            ► BLOCKING CALL 2a ◄  load_demand_profiles()   [dispatch_utils.py:90 → eia_data_io.py:61]
  │            ► BLOCKING CALL 2b ◄  load_generation_profiles()  [dispatch_utils.py:91 → eia_data_io.py:34]
  │            ► BLOCKING CALL 2c ◄  load_fossil_mix()   [dispatch_utils.py:94 → eia_data_io.py:101]
  │            ► BLOCKING CALL 2d ◄  json.load(egrid_emission_rates.json)   [dispatch_utils.py:93]
  │
  │      prints: "[prewarm] {iso}: common data loaded (N cache rows)"
  │
  │      ► BLOCKING CALL 3 ◄  _gas_raw_2025_worst_hour(iso)   [step2_2a_cost_optimization.py:334]
  │         → _expand_missing_archetypes(iso, [mix_info])   [step2_2a_cost_optimization.py:403]
  │             → expand_cache_for_mixes()   [step3a_build_dispatch_cache — local import]
  │                 → reconstruct_hourly_dispatch()   [dispatch_utils.py]
  │                     → _per_resource_dispatch_njit()  ← FIRST-CALL NUMBA JIT  [dispatch_utils.py:569]
  │                     → _battery_loop()               ← FIRST-CALL NUMBA JIT  [dispatch_utils.py:310]
  │                     → _ldes_loop()                  ← FIRST-CALL NUMBA JIT  [dispatch_utils.py:361]
  │
  │      prints: "[prewarm] {iso}: loading EF threshold=..."  (one per threshold)
  │
  │      ► BLOCKING CALL 5 ◄  load_ef_mixes(iso, threshold)  [step2_2a_cost_optimization.py:359]
  │         → step_2_3_pathway_optimizer.load_ef_mixes(iso, threshold)  [step_2_3_pathway_optimizer.py:333]
  │             holds _EF_CACHE_LOCK  [line 345]
  │             → _call_with_timeout(pd.read_parquet(p), 30.0)  [line 355]
  │                 → _TIMEOUT_EXECUTOR.submit(lambda)   [line 257]
  │                 → fut.result(timeout=30.0)            [line 259]
  │
  │      prints: "[prewarm] {iso}: N EF thresholds, N archetypes, N percentiles cached"
  │
  ├─ ThreadPoolExecutor(max_workers=4) spawned   [_sweep_pathways_inproc.py:176]
  │
  │   Each thread → run_pathway(cfg)  → score_ef_batch()
  │
  │      ► BLOCKING CALL 4 ◄  FIRST CALL TO _eval_cost_numba or _batch_eval_and_argmin
  │         [step2_2a_cost_optimization.py:1805 / 2144]
  │         @njit(parallel=True, cache=True)  ← FIRST-CALL NUMBA JIT WITH TBB BACKEND
  │
  │      ► BLOCKING CALL 6 ◄  _call_with_timeout(worst_hour_gas_sizing, 60.0)
  │         [step_2_3_pathway_optimizer.py:965]
  │         → _TIMEOUT_EXECUTOR has max_workers=2
  │         → threads 3 and 4 in the 4-worker outer pool queue behind workers 1 and 2
  │         → if workers 1+2 each take >60 s (numba JIT), threads 3+4 see TimeoutError
  │
  └─ first [solve] line prints inside run_pathway year loop
```

---

## Blocking Call Details

### CALL 1 — Dispatch-cache parquet read  
**File:line:** `dispatch_utils.py:1149` (`pq.read_table(path)`) and `1156–1172` (Python deserialization loop)  
**Called from:** `step2_2a_cost_optimization.py:185` inside `_get_worst_hour_state`, which is invoked at the top of `_ensure_common_data` on the first `prewarm_caches` call.  
**File sizes:** 41 MB (NYISO) to 76 MB (NEISO); mean ~54 MB across 7 ISOs.  
**What it does:** `pq.read_table` reads the entire parquet into an Arrow table (fast, ~1–3 s). The subsequent Python `for i, key in enumerate(keys)` loop at line 1167 materializes each archetype's arrays into a nested `dict[str, ndarray]`. For a 54 MB cache this loop can iterate thousands of archetypes × ~10 fields, each requiring an `ndarray.copy()` — pure Python overhead, 10–60 s, completely silent.  
**Why it hangs:** No timeout wrapper, no progress print, no yield point. Kernel is in `sys.stdin` — the process is alive but unresponsive.  
**Heartbeat (1 line):**  
```python
# dispatch_utils.py, before pq.read_table(path) at line 1149
print(f"[dispatch-cache] {iso}: reading {os.path.getsize(path)/1e6:.1f} MB …", flush=True)
```

---

### CALL 2 — EIA profile deserialization (×3 + JSON)  
**File:line:**  
- `eia_data_io.py:39` — `pd.read_parquet(pq)` for generation profiles (5.6 MB), then triple-nested Python loop (ISOs × years × fuels) calling `.tolist()` on 8760-row slices  
- `eia_data_io.py:67` — demand profiles (3.9 MB), similar nesting  
- `eia_data_io.py:106` — fossil mix (4.7 MB), similar nesting  
- `dispatch_utils.py:92` — `json.load(egrid_emission_rates.json)` (8.8 KB, negligible)  

**Called from:** `dispatch_utils.load_common_data()` (lines 89–94), invoked by `_ensure_common_data` if `demand_data is None` (step2_2a:213–215).  
**Why it hangs:** After `pd.read_parquet` the code runs pure-Python group-by: `for iso in df['iso'].unique()` → `for year in ...` → `for fuel in ...` → `year_df[year_df['fuel'] == fuel]['value'].tolist()`. Each `.tolist()` materializes an 8760-element Python list. No vectorized extraction (e.g., `pivot_table`). For 7 ISOs × 3 years × 5 fuels this is ~105 `.tolist()` calls × 8760 iterations: 5–30 s per file, no output.  
**Heartbeat (1 line per loader):**  
```python
# eia_data_io.py, before pd.read_parquet(pq) in each load_* function
print(f"[eia-io] {name}: reading parquet …", flush=True)
```

---

### CALL 3 — Numba JIT, dispatch kernels (first run, no cache)  
**File:line:**  
- `dispatch_utils.py:310` — `@njit(cache=True)` `_battery_loop`  
- `dispatch_utils.py:361` — `@njit(cache=True)` `_ldes_loop`  
- `dispatch_utils.py:569` — `@njit(cache=True)` `_per_resource_dispatch_njit` (most complex)  

**Called from:** `step3a_build_dispatch_cache.expand_cache_for_mixes` → `dispatch_utils.reconstruct_hourly_dispatch`, triggered by `_expand_missing_archetypes` at `step2_2a_cost_optimization.py:403` inside `_gas_raw_2025_worst_hour` — called at `prewarm_caches:334` immediately after common-data load.  
**No cache found:** `scripts/__pycache__/` is empty; no `.nbi` / `.nbc` files exist. Every `@njit(cache=True)` function compiles from scratch on first call.  
**Why it hangs:** Numba's LLVM backend runs synchronously inside the calling thread with no output to stdout. Compilation time per kernel: `_battery_loop` ~10–20 s, `_ldes_loop` ~10–20 s, `_per_resource_dispatch_njit` ~20–40 s. Combined: 40–80 s of silence after the `[prewarm] … common data loaded` line.  
**Heartbeat (1 line):**  
```python
# step2_2a_cost_optimization.py, before _expand_missing_archetypes at line 403
print(f"[numba] {iso}: warming dispatch kernels (first-compile may take 40–80 s) …", flush=True)
```

---

### CALL 4 — Numba JIT, cost kernels (`parallel=True`), first solve  
**File:line:**  
- `step2_2a_cost_optimization.py:1805` — `@njit(parallel=True, cache=True)` `_eval_cost_numba`  
- `step2_2a_cost_optimization.py:2144` — `@njit(parallel=True, cache=True)` `_batch_eval_and_argmin` (most expensive — tiled batched argmin with `prange`)  

**Called from:** `score_ef_batch()` → first call inside `run_pathway` year loop, reached after prewarm completes when the outer `ThreadPoolExecutor(max_workers=4)` dispatches its first pathway run.  
**Why it hangs:** `parallel=True` forces the TBB (or OpenMP) parallel backend to initialize and link at compile time in addition to LLVM IR compilation. For complex kernels like `_batch_eval_and_argmin` (tiled outer loop + inner `prange` + argmin accumulation), this takes 60–120 s per function. Both functions compile before the first `[solve]` line can print. Combined: 80–240 s of silence. The outer `ThreadPoolExecutor` thread running this appears deadlocked from the user's perspective because it holds no lock and prints nothing.  
**Heartbeat (1 line):**  
```python
# step2_2a_cost_optimization.py, guard before first call to _eval_cost_numba
print("[numba] warming cost evaluation kernels (parallel JIT, ~60–120 s first run) …", flush=True)
```

---

### CALL 5 — EF parquet reads via `_call_with_timeout` (prewarm loop)  
**File:line:** `step_2_3_pathway_optimizer.py:354–357` inside `load_ef_mixes`  
**What it does:** For each needed threshold in the prewarm loop (`step2_2a_cost_optimization.py:356–361`), `load_ef_mixes` acquires `_EF_CACHE_LOCK` (line 345) then submits `pd.read_parquet(p)` to `_TIMEOUT_EXECUTOR` (line 355) and blocks on `fut.result(timeout=30.0)`.  
**File sizes:** CAISO 70% = 34 MB; CAISO 75% = 50 MB (2-part); ERCOT 92.5% = 31 MB.  
**Why it can hang:** The 30 s timeout starts from `fut.result()` call, not from when the lambda starts executing. If `_TIMEOUT_EXECUTOR`'s 2 workers are occupied by concurrent calls (e.g., a slow `worst_hour_gas_sizing`), the submitted lambda queues for up to ~60 s before a worker slot opens, then gets 0 s of the original 30 s budget → `TimeoutError`. In the prewarm path this is single-threaded so it shouldn't stall — but if prewarm runs while another path holds a worker slot, `TimeoutError` propagates out of `load_ef_mixes`, past the `except FileNotFoundError` in the loop (uncaught), and crashes `prewarm_caches` with no diagnostic.  
**Heartbeat:** The existing `[prewarm] loading EF threshold=...` print already provides one line before each call. Add file size to it:  
```python
# step2_2a_cost_optimization.py:357, augment existing print
paths_mb = sum(p.stat().st_size for p in _resolve_ef_paths(iso, threshold)) / 1e6
print(f"[prewarm] {iso}: loading EF threshold={threshold} ({paths_mb:.1f} MB)", flush=True)
```

---

### CALL 6 — `_TIMEOUT_EXECUTOR` starvation (sweep phase, 4 outer threads)  
**File:line:** `step_2_3_pathway_optimizer.py:250–252` (module-level singleton)  
**What it does:** `_TIMEOUT_EXECUTOR = ThreadPoolExecutor(max_workers=2)` is created once at module import. All `_call_with_timeout` calls throughout the module share this 2-worker pool: EF parquet reads (30 s budget), JSON loads (30 s budget), `worst_hour_gas_sizing` (60 s budget).  
**Why it can hang:** The outer sweep runs 4 concurrent threads (`max_workers=4`). If threads 1 and 2 each submit a `worst_hour_gas_sizing` (60 s budget), both `_TIMEOUT_EXECUTOR` workers are occupied. Threads 3 and 4 submit their `worst_hour_gas_sizing` calls; these queue. `fut.result(timeout=60.0)` starts counting. If workers 1 and 2 are in Numba JIT (CALL 4) and take >60 s, threads 3 and 4 raise `TimeoutError` before their tasks even start. Their `run_pathway` calls fail with `TimeoutError` (printed as `FAILED`), not a silent hang — but when 4/4 outer threads serialize through a 2-worker pool that is itself stalled in JIT, the sweep idles for the full JIT duration (~80–240 s) after which all queued tasks start and may immediately timeout.  
**Heartbeat:** Not applicable here — the failure surface is a `TimeoutError` exception, which is visible. The fix is either raising `max_workers` on `_TIMEOUT_EXECUTOR` to ≥ 4 or widening the `worst_hour_gas_sizing` timeout from 60 s to >180 s.

---

## Ordered Diagnosis Checklist

To confirm which site is causing the observed hang, add the single-line heartbeats above in order and re-run. The last heartbeat printed before the hang identifies the culprit.

1. Does `[prewarm] entering prewarm` print?  
   - No → hang is in module-level import (unlikely; no I/O there) or before `main()` runs.  
   - Yes → proceed.

2. Does `[prewarm] common data loaded` print?  
   - No → hang is in **CALL 1** (dispatch-cache parquet) or **CALL 2** (EIA profiles). Add the `[dispatch-cache]` and `[eia-io]` heartbeats.  
   - Yes → proceed.

3. Does the first `[prewarm] loading EF threshold=...` print within ~10 s of `common data loaded`?  
   - No → hang is in **CALL 3** (numba dispatch-kernel JIT). Add the `[numba] warming dispatch kernels` heartbeat.  
   - Yes → proceed; CALL 3 JIT was cached.

4. Does `[prewarm] N EF thresholds … cached` print?  
   - No → hang is in **CALL 5** (EF parquet via `_call_with_timeout`). Check file size, disk speed, or `_TIMEOUT_EXECUTOR` contention.  
   - Yes → prewarm complete; hang is in the solve phase.

5. Does the first `[{iso}] [1/N]` run output appear within ~3 min of prewarm completion?  
   - No → hang is in **CALL 4** (numba cost-kernel parallel JIT, 80–240 s). Add `[numba] warming cost evaluation kernels` heartbeat before first `score_ef_batch` call.  
   - No + runs eventually appear with `FAILED TimeoutError` → **CALL 6** (`_TIMEOUT_EXECUTOR` starvation). Widen `worst_hour_gas_sizing` timeout or raise `max_workers` on `_TIMEOUT_EXECUTOR`.

---

## Files Audited

- `scripts/_sweep_pathways_inproc.py` (full)
- `scripts/step_2_3_pathway_optimizer.py` (lines 1–170, 244–430, 955–1120, 3093–3280)
- `scripts/step2_2a_cost_optimization.py` (lines 1–175, 178–425, 590–670, 1800–2220)
- `scripts/dispatch_utils.py` (lines 1–100, 115–310, 1085–1180)
- `scripts/eia_data_io.py` (full, 120 lines)
- `scripts/pipeline_config.py` (lines 1–80, no I/O at module level)
- `scripts/step3a_build_dispatch_cache.py` (lines 1–80, no I/O at module level)

Data files measured:

| File | Size |
|------|------|
| `data/step3-dispatch/NEISO_dispatch_cache.parquet` | 76 MB |
| `data/step3-dispatch/PJM_dispatch_cache.parquet` | 58 MB |
| `data/step3-dispatch/MISO_dispatch_cache.parquet` | 51 MB |
| `data/step3-dispatch/CAISO_dispatch_cache.parquet` | 54 MB |
| `data/step3-dispatch/ERCOT_dispatch_cache.parquet` | 46 MB |
| `data/step3-dispatch/SPP_dispatch_cache.parquet` | 46 MB |
| `data/step3-dispatch/NYISO_dispatch_cache.parquet` | 41 MB |
| `data/step2.1-ef/step_2_1_EF_CAISO_70.parquet` | 34 MB |
| `data/step2.1-ef/step_2_1_EF_CAISO_75_part*.parquet` | 50 MB (2 parts) |
| `data/eia-930/eia_generation_profiles.parquet` | 5.6 MB |
| `data/eia-930/eia_demand_profiles.parquet` | 3.9 MB |
| `data/eia-930/eia_fossil_mix.parquet` | 4.7 MB |
| `scripts/__pycache__/` | empty — no numba `.nbi`/`.nbc` files |
