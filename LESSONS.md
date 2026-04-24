# LESSONS.md — Durable learnings

> Log a lesson only when it's load-bearing enough to change future behavior AND doesn't fit as a rule in CLAUDE.md. Skip per-session summaries and postmortems — those belong in commit messages. Delete stale entries, don't archive.

## No blended LCOE for a merit-ordered resource — Apr 21, 2026

When a resource has an internal merit-order cost structure (existing baseline → incremental tranches at different $/MWh), the argmin must see per-tranche cost, not a TWh-weighted blend. Blending collapses the "first 1.7 TWh at $25, next 200 TWh at $55, remainder at $70" signal into one number that's equally wrong for every mix, and hides the separate question of whether the existing baseline is being double-charged against a ledger that correctly has it at $0. Fix pattern: decompose to per-tranche (Y, n) TWh + per-year effective $/MWh, sum `Σ tranche_twh × tranche_eff_lcoe` directly, and let the pre-existing ledger convention define which tranches are free. Impact when fixed on ERCOT: ep99 moved from cf=0 collapse to cf=79 (−33% fleet, +100–144% total cost) — directionally correct once the argmin could see the merit-order signal at high CFE thresholds.
