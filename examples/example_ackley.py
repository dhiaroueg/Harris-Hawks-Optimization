# examples/example_ackley.py
import sys
sys.path.append('..')
from hho import HarrisHawksOptimization
import numpy as np

def ackley_function(x):
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e

if __name__ == "__main__":
    print("="*70)
    print("ACKLEY FUNCTION OPTIMIZATION")
    print("="*70)
    
    hho = HarrisHawksOptimization(
        objective_func=ackley_function,
        dim=10,
        lb=-32,
        ub=32,
        n_hawks=30,
        max_iter=500
    )
    
    best_pos, best_fit = hho.optimize(verbose=True)
    print(f"\nBest fitness: {best_fit:.15f}")
    hho.plot_convergence()
