"""
Example: Optimization of Rastrigin Function using HHO
"""

import sys
sys.path.append('..')

from hho import HarrisHawksOptimization
import numpy as np

def rastrigin_function(x):
    """
    Rastrigin Function: Highly multimodal
    Global minimum: f(0,...,0) = 0
    """
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

if __name__ == "__main__":
    print("=" * 70)
    print("RASTRIGIN FUNCTION OPTIMIZATION")
    print("=" * 70)
    
    # Configure HHO
    hho = HarrisHawksOptimization(
        objective_func=rastrigin_function,
        dim=10,
        lb=-5.12,
        ub=5.12,
        n_hawks=30,
        max_iter=500
    )
    
    # Run optimization
    print("\nRunning optimization...")
    best_position, best_fitness = hho.optimize(verbose=True)
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Best position: {best_position}")
    print(f"Best fitness: {best_fitness:.10f}")
    print(f"Expected optimum: 0.0")
    print(f"Error: {abs(best_fitness - 0.0):.10f}")
    
    # Plot convergence
    hho.plot_convergence()
    
    print("\nOptimization completed successfully!")
    print("\nNote: Rastrigin is a challenging multimodal function.")
    print("Getting close to 0 is an excellent result!")
