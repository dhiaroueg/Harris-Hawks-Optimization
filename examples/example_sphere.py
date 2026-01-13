"""
Example: Optimization of Sphere Function using HHO
"""

import sys
sys.path.append('..')

from hho import HarrisHawksOptimization
import numpy as np

def sphere_function(x):
    """Sphere Function: f(x) = sum(x_i^2)"""
    return np.sum(x**2)

if __name__ == "__main__":
    print("="*70)
    print("SPHERE FUNCTION OPTIMIZATION")
    print("="*70)
    
    hho = HarrisHawksOptimization(
        objective_func=sphere_function,
        dim=10,
        lb=-100,
        ub=100,
        n_hawks=30,
        max_iter=500
    )
    
    print("\nRunning optimization...")
    best_position, best_fitness = hho.optimize(verbose=True)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Best position: {best_position}")
    print(f"Best fitness: {best_fitness:.15f}")
    print(f"Expected optimum: 0.0")
    
    hho.plot_convergence()
    print("\nOptimization completed!")
