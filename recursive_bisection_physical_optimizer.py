import numpy as np
import multiprocessing as mp
import pandas as pd
import time

# --- ONTOLOGICAL CONSTANTS (Physics) ---
H = 8760
BATTERY_EFFICIENCY = 0.85
LDES_EFFICIENCY = 0.50
MAX_STORAGE_BOOST = 0.15 # Max theoretical matching gain from 10/10 storage

def fast_score_batch(mix_batch, demand, supply_matrix, pf):
    """
    Speed-Demon: Batch Dot Product via BLAS.
    Checks thousands of mixes in a single CPU operation.
    """
    # Matrix Multiplication: (BatchSize, 5) @ (5, 8760) -> (BatchSize, 8760)
    batch_supply = pf * (mix_batch / 100.0) @ supply_matrix
    
    # Vectorized Matching: Sum the minimum of demand vs supply for each row
    matched = np.minimum(demand, batch_supply).sum(axis=1)
    return matched / demand.sum()

def recursive_worker(mix_chunk, demand, supply_matrix, pf, threshold):
    """The sculptor: Removes the impossible, refines the potential."""
    t_val = threshold / 100.0
    feasible_set = []
    
    # Phase 1: Batch Pruning (The 'Essence' Check)
    raw_scores = fast_score_batch(mix_chunk, demand, supply_matrix, pf)
    
    # Only keep indices that are within striking distance of the target
    survivor_indices = np.where(raw_scores >= (t_val - MAX_STORAGE_BOOST))[0]
    
    # Phase 2: Refined Dispatch (The 'Accident' Check)
    for idx in survivor_indices:
        mix = mix_chunk[idx]
        # Run your specific 7-day rolling LDES and 4hr Battery logic
        final_score = fast_score_with_both_storage(
            demand, supply_matrix, mix/100.0, pf, 10, 10
        )
        
        if final_score >= t_val:
            feasible_set.append({
                'CF': mix[0], 'SOL': mix[1], 'WND': mix[2], 
                'CCS': mix[3], 'HYD': mix[4], 'Score': round(final_score*100, 2)
            })
    return feasible_set

def run_speed_demon_search(iso, threshold, procurement_pct):
    # Prepare Physics Assets (Assume these are pre-loaded from your data)
    demand_arr, supply_matrix, h_cap = prepare_physics_assets(iso)
    pf = procurement_pct / 100.0
    
    # 1. Generate the 1% Grid (Barycentric Simplex)
    # This generates ~4.6M combinations in <1 second
    grid = [
        [cf, sol, wnd, ccs, 100 - (cf + sol + wnd + ccs)]
        for cf in range(0, 101)
        for sol in range(0, 101 - cf)
        for wnd in range(0, 101 - cf - sol)
        for ccs in range(0, 101 - cf - sol - wnd)
        if 0 <= (100 - (cf + sol + wnd + ccs)) <= h_cap
    ]
    grid = np.array(grid, dtype=np.float32)
    
    # 2. Parallel Dispatch
    num_cores = mp.cpu_count()
    chunks = np.array_split(grid, num_cores * 10) # Smaller chunks for better load balancing
    
    print(f"Recursively scrubbing {len(grid)} combinations at 1% intervals...")
    start = time.time()
    
    with mp.Pool(processes=num_cores) as pool:
        arg_list = [(chunk, demand_arr, supply_matrix, pf, threshold) for chunk in chunks]
        results = pool.starmap(recursive_worker, arg_list)
        
    # 3. Flatten and Export
    flattened = [item for sublist in results for item in sublist]
    df = pd.DataFrame(flattened)
    
    print(f"Teleology achieved in {time.time() - start:.1f}s.")
    print(f"Found {len(df)} feasible physical combinations.")
    
    df.to_csv(f"{iso}_{threshold}pct_feasibility_map.csv", index=False)
    return df
