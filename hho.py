import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class HarrisHawksOptimization:
    """
    Implémentation complète de l'algorithme Harris Hawks Optimization (HHO)
    
    Paramètres:
    -----------
    objective_func : fonction
        La fonction objectif à minimiser
    dim : int
        Dimension du problème (nombre de variables)
    lb : array ou float
        Borne inférieure de l'espace de recherche
    ub : array ou float
        Borne supérieure de l'espace de recherche
    n_hawks : int
        Nombre de faucons (taille de la population)
    max_iter : int
        Nombre maximum d'itérations
    """
    
    def __init__(self, objective_func, dim, lb, ub, n_hawks=30, max_iter=500):
        self.objective_func = objective_func
        self.dim = dim
        self.lb = np.array([lb] * dim) if isinstance(lb, (int, float)) else np.array(lb)
        self.ub = np.array([ub] * dim) if isinstance(ub, (int, float)) else np.array(ub)
        self.n_hawks = n_hawks
        self.max_iter = max_iter
        
        # Historique pour le suivi
        self.convergence_curve = []
        self.best_position_history = []
        
    def initialize_population(self):
        """
        ÉTAPE 1: INITIALISATION DE LA POPULATION
        ==========================================
        Génère aléatoirement les positions initiales des faucons
        dans l'espace de recherche défini par [lb, ub]
        """
        population = np.zeros((self.n_hawks, self.dim))
        for i in range(self.n_hawks):
            for j in range(self.dim):
                population[i, j] = self.lb[j] + (self.ub[j] - self.lb[j]) * np.random.rand()
        return population
    
    def evaluate_fitness(self, population):
        """
        ÉTAPE 2: ÉVALUATION DE LA FITNESS
        ==================================
        Calcule la valeur de la fonction objectif pour chaque faucon
        """
        fitness = np.zeros(self.n_hawks)
        for i in range(self.n_hawks):
            fitness[i] = self.objective_func(population[i, :])
        return fitness
    
    def levy_flight(self, dim):
        """
        VOL DE LÉVY
        ===========
        Génère un mouvement basé sur la distribution de Lévy
        Utilisé pour améliorer l'exploration et éviter les optimums locaux
        
        Le vol de Lévy simule des mouvements biologiques naturels:
        - Alternance de petits pas et de grands sauts
        - Améliore la capacité d'exploration
        """
        import math
        beta = 1.5
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        
        u = np.random.randn(dim) * sigma
        v = np.random.randn(dim)
        step = u / np.abs(v)**(1 / beta)
        
        return step
    
    def explore(self, population, rabbit_position, t):
        """
        ÉTAPE 3: PHASE D'EXPLORATION (|E| >= 1)
        ========================================
        Les faucons cherchent la proie en explorant l'espace de recherche
        
        Deux stratégies:
        q >= 0.5: Position basée sur d'autres faucons et la proie
        q < 0.5:  Position aléatoire dans l'espace de recherche
        """
        new_population = np.copy(population)
        
        for i in range(self.n_hawks):
            q = np.random.rand()
            rand_hawk_idx = np.random.randint(0, self.n_hawks)
            
            if q >= 0.5:
                # Stratégie 1: Se baser sur un faucon aléatoire et la proie
                r1 = np.random.rand()
                r2 = np.random.rand()
                new_population[i, :] = (population[rand_hawk_idx, :] - 
                                       r1 * abs(population[rand_hawk_idx, :] - 
                                       2 * r2 * population[i, :]))
            else:
                # Stratégie 2: Position aléatoire dans l'espace
                r3 = np.random.rand()
                r4 = np.random.rand()
                new_population[i, :] = (rabbit_position - 
                                       population.mean(axis=0) - 
                                       r3 * (self.lb + r4 * (self.ub - self.lb)))
            
            # Vérification des bornes
            new_population[i, :] = np.clip(new_population[i, :], self.lb, self.ub)
        
        return new_population
    
    def soft_besiege(self, hawk_position, rabbit_position, E):
        """
        EXPLOITATION - Stratégie 1: SIÈGE DOUX (Soft Besiege)
        ======================================================
        Conditions: |E| >= 0.5 et r >= 0.5
        La proie a encore de l'énergie et tente de s'échapper
        Les faucons encerclent progressivement
        """
        delta_E = rabbit_position - hawk_position
        new_position = delta_E - E * abs(np.random.rand() * rabbit_position - hawk_position)
        return new_position
    
    def hard_besiege(self, hawk_position, rabbit_position, E):
        """
        EXPLOITATION - Stratégie 2: SIÈGE DUR (Hard Besiege)
        ====================================================
        Conditions: |E| < 0.5 et r >= 0.5
        La proie est épuisée, attaque directe et rapide
        """
        new_position = rabbit_position - E * abs(rabbit_position - hawk_position)
        return new_position
    
    def soft_besiege_progressive(self, hawk_position, rabbit_position, E, dim):
        """
        EXPLOITATION - Stratégie 3: SIÈGE DOUX AVEC MANŒUVRES PROGRESSIVES
        ==================================================================
        Conditions: |E| >= 0.5 et r < 0.5
        La proie tente de s'échapper, les faucons font des manœuvres tactiques
        Utilise le vol de Lévy pour des mouvements imprévisibles
        """
        jump_strength = 2 * (1 - np.random.rand())
        Y = rabbit_position - E * abs(jump_strength * rabbit_position - hawk_position)
        
        # Si la nouvelle position n'est pas meilleure, tenter un vol de Lévy
        if self.objective_func(Y) >= self.objective_func(hawk_position):
            S = np.random.rand(dim) * hawk_position
            LF = self.levy_flight(dim)
            Z = Y + S * LF
            return np.clip(Z, self.lb, self.ub)
        else:
            return np.clip(Y, self.lb, self.ub)
    
    def hard_besiege_progressive(self, hawk_position, rabbit_position, E, dim, population):
        """
        EXPLOITATION - Stratégie 4: SIÈGE DUR AVEC MANŒUVRES PROGRESSIVES
        =================================================================
        Conditions: |E| < 0.5 et r < 0.5
        La proie est épuisée mais tente encore de fuir
        Les faucons exécutent une attaque coordonnée finale
        """
        jump_strength = 2 * (1 - np.random.rand())
        avg_position = population.mean(axis=0)
        
        Y = rabbit_position - E * abs(jump_strength * rabbit_position - avg_position)
        
        # Si la nouvelle position n'est pas meilleure, tenter un vol de Lévy
        if self.objective_func(Y) >= self.objective_func(hawk_position):
            S = np.random.rand(dim) * hawk_position
            LF = self.levy_flight(dim)
            Z = Y + S * LF
            return np.clip(Z, self.lb, self.ub)
        else:
            return np.clip(Y, self.lb, self.ub)
    
    def exploit(self, population, rabbit_position, E):
        """
        ÉTAPE 4: PHASE D'EXPLOITATION (|E| < 1)
        ========================================
        Les faucons attaquent la proie selon 4 stratégies différentes
        Le choix dépend de l'énergie E et de la capacité d'évasion r
        """
        new_population = np.copy(population)
        
        for i in range(self.n_hawks):
            r = np.random.rand()  # Probabilité d'évasion de la proie
            
            if abs(E) >= 0.5 and r >= 0.5:
                # Stratégie 1: Soft besiege
                new_population[i, :] = self.soft_besiege(
                    population[i, :], rabbit_position, E
                )
                
            elif abs(E) >= 0.5 and r < 0.5:
                # Stratégie 2: Soft besiege avec manœuvres progressives
                new_population[i, :] = self.soft_besiege_progressive(
                    population[i, :], rabbit_position, E, self.dim
                )
                
            elif abs(E) < 0.5 and r >= 0.5:
                # Stratégie 3: Hard besiege
                new_population[i, :] = self.hard_besiege(
                    population[i, :], rabbit_position, E
                )
                
            else:  # abs(E) < 0.5 and r < 0.5
                # Stratégie 4: Hard besiege avec manœuvres progressives
                new_population[i, :] = self.hard_besiege_progressive(
                    population[i, :], rabbit_position, E, self.dim, population
                )
            
            # Vérification des bornes
            new_population[i, :] = np.clip(new_population[i, :], self.lb, self.ub)
        
        return new_population
    
    def optimize(self, verbose=True):
        """
        BOUCLE PRINCIPALE D'OPTIMISATION
        =================================
        Orchestre toutes les étapes de l'algorithme HHO
        """
        # Initialisation
        population = self.initialize_population()
        fitness = self.evaluate_fitness(population)
        
        # Trouver le meilleur faucon initial (la proie - rabbit)
        best_idx = np.argmin(fitness)
        rabbit_position = population[best_idx, :].copy()
        rabbit_fitness = fitness[best_idx]
        
        # E0: Énergie initiale de la proie (aléatoire entre -1 et 1)
        E0 = 2 * np.random.rand() - 1
        
        # Boucle principale
        for t in range(self.max_iter):
            # CALCUL DE L'ÉNERGIE DE LA PROIE
            # E diminue linéairement de E0 à 0
            E = 2 * E0 * (1 - t / self.max_iter)
            
            # DÉCISION: EXPLORATION OU EXPLOITATION
            if abs(E) >= 1:
                # Phase d'exploration
                population = self.explore(population, rabbit_position, t)
            else:
                # Phase d'exploitation
                population = self.exploit(population, rabbit_position, E)
            
            # Évaluer la nouvelle population
            fitness = self.evaluate_fitness(population)
            
            # Mettre à jour la meilleure solution (rabbit)
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < rabbit_fitness:
                rabbit_position = population[best_idx, :].copy()
                rabbit_fitness = fitness[best_idx]
            
            # Enregistrer l'historique
            self.convergence_curve.append(rabbit_fitness)
            self.best_position_history.append(rabbit_position.copy())
            
            # Affichage
            if verbose and (t % 50 == 0 or t == self.max_iter - 1):
                print(f"Itération {t+1}/{self.max_iter} - Meilleure fitness: {rabbit_fitness:.6f}")
        
        return rabbit_position, rabbit_fitness
    
    def plot_convergence(self):
        """
        Visualisation de la courbe de convergence
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_curve, linewidth=2, color='#2E86AB')
        plt.xlabel('Itération', fontsize=12)
        plt.ylabel('Meilleure Fitness', fontsize=12)
        plt.title('Courbe de Convergence - Harris Hawks Optimization', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ============================================================================
# FONCTIONS DE TEST BENCHMARK
# ============================================================================

def sphere_function(x):
    """
    Fonction Sphère: f(x) = sum(x_i^2)
    Minimum global: f(0,...,0) = 0
    """
    return np.sum(x**2)

def rastrigin_function(x):
    """
    Fonction Rastrigin: Multimodale avec nombreux minima locaux
    Minimum global: f(0,...,0) = 0
    """
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock_function(x):
    """
    Fonction Rosenbrock: Vallée étroite, difficile à optimiser
    Minimum global: f(1,...,1) = 0
    """
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def ackley_function(x):
    """
    Fonction Ackley: Multimodale avec un grand nombre de minima locaux
    Minimum global: f(0,...,0) = 0
    """
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HARRIS HAWKS OPTIMIZATION - DÉMONSTRATION")
    print("=" * 70)
    
    # Configuration du problème
    dimension = 10
    lower_bound = -10
    upper_bound = 10
    n_hawks = 30
    max_iterations = 500
    
    # Test sur la fonction Sphere
    print("\n1. Test sur la fonction Sphere")
    print("-" * 70)
    hho_sphere = HarrisHawksOptimization(
        objective_func=sphere_function,
        dim=dimension,
        lb=lower_bound,
        ub=upper_bound,
        n_hawks=n_hawks,
        max_iter=max_iterations
    )
    
    best_pos_sphere, best_fit_sphere = hho_sphere.optimize()
    print(f"\nRésultat final:")
    print(f"Meilleure position: {best_pos_sphere}")
    print(f"Meilleure fitness: {best_fit_sphere:.10f}")
    hho_sphere.plot_convergence()
    
    # Test sur la fonction Rastrigin
    print("\n2. Test sur la fonction Rastrigin")
    print("-" * 70)
    hho_rastrigin = HarrisHawksOptimization(
        objective_func=rastrigin_function,
        dim=dimension,
        lb=-5.12,
        ub=5.12,
        n_hawks=n_hawks,
        max_iter=max_iterations
    )
    
    best_pos_rastrigin, best_fit_rastrigin = hho_rastrigin.optimize()
    print(f"\nRésultat final:")
    print(f"Meilleure position: {best_pos_rastrigin}")
    print(f"Meilleure fitness: {best_fit_rastrigin:.10f}")
    hho_rastrigin.plot_convergence()
    
    print("\n" + "=" * 70)
    print("OPTIMISATION TERMINÉE")
    print("=" * 70)