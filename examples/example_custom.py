"""
Example: Custom Optimization Problem using HHO
Demonstrates how to use HHO with your own objective function
"""

import sys
sys.path.append('..')

from hho import HarrisHawksOptimization
import numpy as np

def custom_function(x):
    """
    Custom Function Example
    You can replace this with ANY function you want to optimize!
    
    This example: Minimize sum of squared differences from target
    """
    target = np.array([5, 3, -2, 7, 1])
    return np.sum((x - target)**2)

if __name__ == "__main__":
    print("=" * 70)
    print("CUSTOM FUNCTION OPTIMIZATION")
    print("=" * 70)
    
    print("\nObjective: Find x that minimizes (x - target)^2")
    print("Target: [5, 3, -2, 7, 1]")
    
    # Configure HHO
    hho = HarrisHawksOptimization(
        objective_func=custom_function,
        dim=5,
        lb=-10,
        ub=10,
        n_hawks=20,
        max_iter=300
    )
    
    # Run optimization
    print("\nRunning optimization...")
    best_position, best_fitness = hho.optimize(verbose=True)
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Best solution found: {best_position}")
    print(f"Target solution:     [5, 3, -2, 7, 1]")
    print(f"Best fitness: {best_fitness:.10f}")
    print(f"Expected optimum: 0.0")
    
    # Plot convergence
    hho.plot_convergence()
    
    print("\n" + "=" * 70)
    print("HOW TO USE WITH YOUR OWN PROBLEM")
    print("=" * 70)
    print("1. Define your objective function")
    print("2. Set appropriate bounds (lb, ub)")
    print("3. Choose population size and iterations")
    print("4. Run optimize() and get results!")
    print("\nIt's that simple! 🎉")
