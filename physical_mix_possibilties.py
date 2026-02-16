import numpy as np
import multiprocessing as mp

def generate_simplex_grid(step_size=1):
    """
    The quickest way to generate a 1% grid.
    Calculates only valid coordinates that sum to 100.
    """
    # Using a list comprehension over a generator for speed at 1%
    return [
        [cf, sol, wnd, ccs, 100 - (cf + sol + wnd + ccs)]
        for cf in range(0, 101, step_size)
        for sol in range(0, 101 - cf, step_size)
        for wnd in range(0, 101 - cf - sol, step_size)
        for ccs in range(0, 101 - cf - sol - wnd, step_size)
    ]

def fast_pruning_worker(mix_chunk, demand, supply_matrix, pf, threshold):
    """
    Pure NumPy vectorized 'Essence' check.
    This is the filter that makes the 1% grid viable.
    """
    t_val = threshold / 100.0
    # Batch dot product: (N, 5) @ (5, 8760) -> (N, 8760)
    # This checks thousands of mixes simultaneously in a single CPU cycle
    batch_supply = pf * (np.array(mix_chunk) / 100.0) @ supply_matrix
    
    # Vectorized score: (N, 8760) compared to (8760,)
    # We only keep indices that pass the 'Potentiality' check
    raw_scores = np.minimum(demand, batch_supply).sum(axis=1) / demand.sum()
    
    # Only run the heavy storage logic on the remaining 'survivors'
    survivors = np.where(raw_scores >= (t_val - 0.15))[0]
    
    results = []
    for idx in survivors:
        # Heavylifting storage logic only happens here
        final_score = fast_score_with_both_storage(demand, supply_matrix, mix_chunk[idx]/100.0, pf, 10, 10)
        if final_score >= t_val:
            results.append(mix_chunk[idx])
            
    return results
