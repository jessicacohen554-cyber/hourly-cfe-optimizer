# Market Simulator Peer Review — Implementation Guide

**Date**: 2026-03-21
**File**: `market-simulator/scripts/market_simulation.py` (4,800+ lines)
**Branch**: `claude/review-market-simulator-Frzi0`

## Status

- [x] **Rename** `run_sweep_405.py` → `run_sweep_1215.py` (done, pushed)
- [ ] **Fix #1**: `_is_nuclear_plant()` operator precedence
- [ ] **Fix #2**: Vectorize `compute_storage_arbitrage_from_lmp()`
- [ ] **Fix #3**: `_scenario_weight()` dead code cleanup
- [ ] **Fix #4**: `_retire_nuclear_plants()` cumulative_retired_mw tracking
- [ ] **Fix #5**: `cumulative_gw` shared-across-ISOs documentation
- [ ] **Fix #6**: Skip `_provenance` in `aggregate_sweep_percentiles()`
- [ ] **Fix #7**: Add logging to zonal LMP exception swallowing
- [ ] **Refactor #8**: Extract LMP computation helper (deduplicate Pass 1/Pass 2)
- [ ] **Refactor #9**: Dataclass returns for `compute_lmp_at_threshold()` (11-tuple)
- [ ] **Refactor #10**: Dataclass returns for `compute_market_deployment()` (9-tuple)

---

## Fix #1: `_is_nuclear_plant()` operator precedence (line ~1194)

**Bug**: `and` binds tighter than `or`, making the third clause redundant dead code.

**Current** (line 1194):
```python
return ('NUC' in fuel or 'UR' in fuel or
        'ST' == mover and 'NUC' in fuel or
        utype == 'nuclear')
```

**Replace with**:
```python
return ('NUC' in fuel or 'UR' in fuel or
        (mover == 'ST' and 'NUC' in fuel) or
        utype == 'nuclear')
```

**Why**: Adds explicit parentheses for intent clarity and fixes the `==` direction (`'ST' == mover` → `mover == 'ST'` for readability, Yoda conditions are discouraged).

---

## Fix #2: Vectorize `compute_storage_arbitrage_from_lmp()` (line ~1659)

**Problem**: Python for-loop over 365 windows per tech. Called ~51K times in a full sweep. Adds ~25s of wall time.

**Current** (lines 1684–1700):
```python
for w in range(n_windows):
    w_start = w * window_hours
    w_end = min(w_start + window_hours, H)
    if w_end - w_start < 2 * charge_hours_per_window:
        continue
    window_lmp = lmp[w_start:w_end]
    sorted_idx = np.argsort(window_lmp)
    charge_cost = np.sum(window_lmp[sorted_idx[:charge_hours_per_window]])
    discharge_rev = np.sum(window_lmp[sorted_idx[-discharge_hours_per_window:]])
    cycle_revenue = discharge_rev * rte - charge_cost
    total_revenue_dollar_per_mw += cycle_revenue
```

**Replace entire for-loop block** (from `total_revenue_dollar_per_mw = 0.0` through `total_revenue_dollar_per_mw += cycle_revenue`) **with**:
```python
total_revenue_dollar_per_mw = 0.0
charge_hours_per_window = min(duration, window_hours // 2)
discharge_hours_per_window = charge_hours_per_window

# Vectorized: reshape LMP into (n_windows, window_hours) matrix
# Truncate to exact multiple of window_hours for clean reshape
usable_hours = n_windows * window_hours
if usable_hours > H:
    usable_hours = (H // window_hours) * window_hours
    n_windows = usable_hours // window_hours
if n_windows <= 0 or usable_hours <= 0:
    results[tech] = 0.0
    continue

lmp_matrix = lmp[:usable_hours].reshape(n_windows, window_hours)

# Check minimum window size (need at least 2 × charge_hours)
if window_hours < 2 * charge_hours_per_window:
    results[tech] = 0.0
    continue

# Sort each window independently along axis=1
sorted_matrix = np.sort(lmp_matrix, axis=1)

# Charge at cheapest hours (left columns), discharge at most expensive (right columns)
charge_cost_per_window = sorted_matrix[:, :charge_hours_per_window].sum(axis=1)
discharge_rev_per_window = sorted_matrix[:, -discharge_hours_per_window:].sum(axis=1)

# Net revenue per cycle: discharge × RTE - charge
cycle_revenues = discharge_rev_per_window * rte - charge_cost_per_window
total_revenue_dollar_per_mw = float(np.sum(np.maximum(cycle_revenues, 0.0)))
```

**Why**: Replaces N iterations of `np.argsort` (O(W log W) each) with a single `np.sort` on a 2D matrix (same total work but no Python loop overhead). Expected ~10-20x speedup. Note: using `np.maximum(cycle_revenues, 0.0)` before summing preserves the `max(0.0, ...)` applied to the final result, but does it per-window which is slightly more conservative (avoids negative windows canceling positive ones — matches real storage operator behavior where you wouldn't discharge at a loss).

**Behavioral note**: The original sums all cycle revenues (including negative) then floors at 0. The vectorized version floors per-window. This is arguably *more correct* — a storage operator wouldn't dispatch in a window where revenue is negative. If exact backward compatibility is required, change the last line to:
```python
total_revenue_dollar_per_mw = float(np.sum(cycle_revenues))
```

---

## Fix #3: `_scenario_weight()` dead code cleanup (lines ~4548–4563)

**Problem**: Lines 4548-4554 assign variables that are immediately overwritten at lines 4558-4563.

**Current**:
```python
demand_code = parts[1]
ppa_code = parts[-3]
gas_code = parts[-2]
# The queue_code and nfc_code are the last two? Re-check format:
# MKT_{D}_{price_name}_{P}_{G}_{Q}_{N}  → 7 minimum tokens
queue_code = parts[-2]
nfc_code = parts[-1]

# Actually re-parse: parts = [MKT, D, ...price_name..., P, G, Q, N]
# Last 4 single-char codes: ppa, gas, queue, nfc
nfc_code = parts[-1]
queue_code = parts[-2]
gas_code = parts[-3]
ppa_code = parts[-4]
# price_name is everything between index 2 and -4
price_name = '_'.join(parts[2:-4])
```

**Replace with**:
```python
demand_code = parts[1]
# Scenario ID format: MKT_{D}_{price_name}_{P}_{G}_{Q}_{N}
# Last 4 single-char codes: ppa, gas_friction, queue, new_fossil_cost
# price_name occupies all tokens between index 2 and -4 (may contain underscores)
nfc_code = parts[-1]
queue_code = parts[-2]
gas_code = parts[-3]
ppa_code = parts[-4]
price_name = '_'.join(parts[2:-4])
```

---

## Fix #4: `_retire_nuclear_plants()` — cumulative tracking (line ~1278)

**Problem**: `cumulative_retired_mw` is a parameter that's modified locally but not returned, so the caller never sees the updated total.

**Current** (line 1231 signature):
```python
def _retire_nuclear_plants(iso, year, state, active_plants, nuclear_contract,
                           cumulative_retired_mw, max_retirable_mw, _log=print):
```

**No code change needed** — the function is always called *after* the fossil retirement loop completes (line 1155), so the nuclear retirements don't need to feed back into the fossil loop. However, the `cumulative_retired_mw` check inside `_retire_nuclear_plants` (line 1273) uses the stale value, meaning the **nuclear** reliability floor check doesn't account for earlier nuclear retirements in the same call.

**Fix** (line 1278): Change from:
```python
        cumulative_retired_mw += cap_mw
```
to — no fix needed at the caller level, but the internal tracking is already correct since it only accumulates within nuclear plants. The real risk is if both nuclear and fossil retire in the same year and breach the floor. To fully fix:

In `_apply_plant_level_retirement()`, change the call at line 1155-1157:
```python
    nuclear_retired_plants = _retire_nuclear_plants(
        iso, year, state, active_plants, nuclear_contract,
        cumulative_retired_this_year, max_retirable_mw, _log)
```
This already passes `cumulative_retired_this_year` (which includes fossil retirements). The function correctly adds to its local copy. The only issue is that multiple nuclear retirements within the same `_retire_nuclear_plants` call DO properly accumulate (line 1278 updates the local `cumulative_retired_mw`). **This is actually fine — no code change needed.** Downgrading to documentation-only.

---

## Fix #5: `cumulative_gw` shared across ISOs — add documentation (line ~3603)

**Current**:
```python
cumulative_gw = dict(WRIGHT_CUMULATIVE_GW_2025)
```

**Add comment**:
```python
# Global cumulative GW — shared across all ISOs (intentional: represents worldwide
# technology learning. ISO-specific deployments contribute to global cost reductions
# via Wright's Law. This is the standard assumption in IEA WEO / IRENA models.)
cumulative_gw = dict(WRIGHT_CUMULATIVE_GW_2025)
```

---

## Fix #6: Skip `_provenance` in `aggregate_sweep_percentiles()` (line ~4639)

**Problem**: Inner loop iterates over provenance metadata keys (strings), calling `.get('year')` on non-dict values.

**Current**:
```python
    for scenario_id, iso_results in all_results.items():
        for iso, year_results in iso_results.items():
```

**Replace with**:
```python
    for scenario_id, iso_results in all_results.items():
        if scenario_id.startswith('_'):
            continue  # Skip _provenance and other metadata keys
        if not isinstance(iso_results, dict):
            continue
        for iso, year_results in iso_results.items():
```

---

## Fix #7: Add logging to zonal LMP exception swallowing (line ~581)

**Current**:
```python
    except Exception:
        # Fall back to copper-plate
        pass
```

**Replace with**:
```python
    except Exception as _zonal_err:
        # Fall back to copper-plate LMP — log the failure for debugging
        logger.debug("Zonal LMP failed for %s (clean=%.1f%%): %s — using copper-plate",
                     iso, clean_pct, _zonal_err)
```

---

## Refactor #8: Extract LMP computation helper (lines ~3805–3884)

**Problem**: 30 lines of identical code duplicated between LMP Pass 1 and Pass 2, differing only in storage pct values and cache key.

**Add new helper function** (insert before `run_market_simulation`, around line ~3516):
```python
def _compute_or_cache_lmp(iso, current_pct, conditions, demand_norm, demand_mw_profile,
                           supply_profiles_iso, resource_pcts, storage_pcts,
                           ic_norm, ic_firm_mw, dr_level, growth_factor, state,
                           year, carbon_price, interchange_enabled, nb_bucket,
                           lmp_cache=None):
    """Compute LMP at threshold, with caching. Returns LmpResult tuple."""
    stor_key = tuple(sorted(storage_pcts.items())) if storage_pcts else ()
    cache_key = (iso, current_pct, conditions['fuel_level'],
                 conditions['demand_growth'], year, carbon_price,
                 interchange_enabled, dr_level, nb_bucket, stor_key)

    if lmp_cache is not None and cache_key in lmp_cache:
        return lmp_cache[cache_key]

    result = compute_lmp_at_threshold(
        iso, current_pct, conditions['fuel_level'],
        demand_norm, demand_mw_profile,
        supply_profiles_iso, resource_pcts,
        battery_pct=storage_pcts.get('battery', 0),
        battery8_pct=storage_pcts.get('battery8', 0),
        ldes_pct=storage_pcts.get('ldes', 0),
        h2_pct=storage_pcts.get('h2', 0),
        carbon_price=carbon_price,
        nox_price=conditions.get('nox_price', 0.0),
        sox_price=conditions.get('sox_price', 0.0),
        nox_limit=conditions.get('nox_limit'),
        sox_limit=conditions.get('sox_limit'),
        custom_fuel_prices=conditions.get('custom_fuel_prices'),
        custom_co2_price=conditions.get('custom_co2_price'),
        custom_heat_rates=conditions.get('custom_heat_rates'),
        custom_vom=conditions.get('custom_vom'),
        interchange_norm=ic_norm,
        firm_import_mw=ic_firm_mw,
        dr_level=dr_level,
        demand_growth_factor=growth_factor,
        new_fossil_builds=state.get('new_fossil_builds'),
    )

    if lmp_cache is not None:
        lmp_cache[cache_key] = result

    return result
```

**Then replace Pass 1** (lines ~3805-3833) with:
```python
            lmp_result = _compute_or_cache_lmp(
                iso, current_pct, conditions, demand_norm, demand_mw_profile,
                supply_profiles_iso, resource_pcts, _prev_stor,
                ic_norm, ic_firm_mw, dr_level, growth_factor, state,
                year, carbon_price, interchange_enabled, _nb_bucket,
                lmp_cache=_lmp_cache)
            (hourly_lmp, avg_lmp, p90_lmp, gen_econ, dr_metrics,
             zonal_congestion_data, scarcity_hours_frac, _zonal_lmp_matrix,
             _zonal_zone_names, curtailment_rate, lmp_confidence) = lmp_result
```

**And replace Pass 2** (lines ~3857-3884) with:
```python
                lmp_result = _compute_or_cache_lmp(
                    iso, current_pct, conditions, demand_norm, demand_mw_profile,
                    supply_profiles_iso, resource_pcts, _new_stor,
                    ic_norm, ic_firm_mw, dr_level, growth_factor, state,
                    year, carbon_price, interchange_enabled, _nb_bucket,
                    lmp_cache=_lmp_cache)
                (hourly_lmp, avg_lmp, p90_lmp, gen_econ, dr_metrics,
                 zonal_congestion_data, scarcity_hours_frac, _zonal_lmp_matrix,
                 _zonal_zone_names, curtailment_rate, lmp_confidence) = lmp_result
```

**Delete** the now-unused `_lmp_key` / `_lmp_key2` construction and inline cache logic blocks.

---

## Refactor #9: Dataclass for `compute_lmp_at_threshold()` return (line ~693)

**Add dataclass** (after imports, ~line 120):
```python
@dataclass
class LmpResult:
    """Return type for compute_lmp_at_threshold()."""
    hourly_lmp: np.ndarray
    avg_lmp: float
    lmp_p90: float
    gen_econ: dict
    dr_metrics: dict
    zonal_stats: Optional[dict]
    scarcity_hours_fraction: float
    zonal_lmp_matrix: Optional[np.ndarray]
    zonal_zone_names: Optional[list]
    curtailment_rate: float
    lmp_confidence: float
```

**Change return** at line ~693 from:
```python
return hourly_lmp, avg_lmp, p90_lmp, gen_econ, dr_metrics, zonal_stats, scarcity_hours_fraction, zonal_lmp_matrix, zonal_zone_names, curtailment_rate, lmp_confidence
```
to:
```python
return LmpResult(
    hourly_lmp=hourly_lmp, avg_lmp=avg_lmp, lmp_p90=p90_lmp,
    gen_econ=gen_econ, dr_metrics=dr_metrics, zonal_stats=zonal_stats,
    scarcity_hours_fraction=scarcity_hours_fraction,
    zonal_lmp_matrix=zonal_lmp_matrix, zonal_zone_names=zonal_zone_names,
    curtailment_rate=curtailment_rate, lmp_confidence=lmp_confidence,
)
```

**Update all callers** to use attribute access (`result.avg_lmp`) instead of tuple unpacking. The `_compute_or_cache_lmp` helper (Refactor #8) can return `LmpResult` directly, and the destructuring in `run_market_simulation` changes to:
```python
lmp_result = _compute_or_cache_lmp(...)
hourly_lmp = lmp_result.hourly_lmp
avg_lmp = lmp_result.avg_lmp
# ... etc, or just use lmp_result.avg_lmp inline
```

**Backward compat**: Since `LmpResult` is a dataclass, existing tuple unpacking like `a, b, c, ... = compute_lmp_at_threshold(...)` will break. To support both patterns during migration, add `__iter__` to the dataclass:
```python
    def __iter__(self):
        return iter((self.hourly_lmp, self.avg_lmp, self.lmp_p90,
                     self.gen_econ, self.dr_metrics, self.zonal_stats,
                     self.scarcity_hours_fraction, self.zonal_lmp_matrix,
                     self.zonal_zone_names, self.curtailment_rate,
                     self.lmp_confidence))
```

This lets existing tuple unpacking work unchanged while new code can use named attributes.

---

## Refactor #10: Dataclass for `compute_market_deployment()` return (line ~3040)

**Add dataclass** (after `LmpResult`):
```python
@dataclass
class DeploymentResult:
    """Return type for compute_market_deployment()."""
    new_clean_pct: float
    deployed: Dict[str, float]            # {resource: twh_deployed}
    zone_results: List[dict]
    rev_breakdown: dict
    blended_cost: float
    blended_revenue: float
    remaining_gw: float
    energy_rev_by_resource: Dict[str, float]
    capture_rates: Dict[str, float]

    def __iter__(self):
        return iter((self.new_clean_pct, self.deployed, self.zone_results,
                     self.rev_breakdown, self.blended_cost, self.blended_revenue,
                     self.remaining_gw, self.energy_rev_by_resource,
                     self.capture_rates))
```

**Change return** at line ~3040 from the tuple to:
```python
return DeploymentResult(
    new_clean_pct=round(clean_pct, 2),
    deployed=deployed,
    zone_results=zone_results,
    rev_breakdown=rev_breakdown,
    blended_cost=round(blended_cost, 2),
    blended_revenue=round(blended_revenue, 2),
    remaining_gw=remaining_gw,
    energy_rev_by_resource=energy_rev_by_res,
    capture_rates=capture_rates,
)
```

---

## Execution Order

These fixes are independent and can be applied in any order. Recommended:

1. **Fixes #1, #3, #5, #6, #7** — trivial, no risk (5 min total)
2. **Refactor #9, #10** — dataclasses with `__iter__` for backward compat (10 min)
3. **Refactor #8** — LMP helper extraction (15 min, depends on #9)
4. **Fix #2** — storage vectorization (10 min, test with single-ISO sweep)

## Testing

After all changes:
```bash
cd market-simulator/scripts
python -c "import py_compile; py_compile.compile('market_simulation.py', doraise=True)"
python -c "from market_simulation import build_market_scenarios; print(f'{len(build_market_scenarios())} scenarios')"
python market_simulation.py --single --isos CAISO  # Quick smoke test
```
