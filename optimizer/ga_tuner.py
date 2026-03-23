"""
Genetic Algorithm for L1 Controller Parameter Optimization
"""

import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import warnings
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plant.rov_dynamics import ROVDynamics
from plant.environment import OceanCurrent
from controller.l1_adaptive import L1AdaptiveControllerStable as L1AdaptiveController


class GATuner:
    """Genetic Algorithm tuner for L1 controller parameters"""
    
    def __init__(self, bounds, pop_size=50, n_generations=30, 
                 crossover_prob=0.8, mutation_prob=0.2):
        """
        Initialize GA tuner
        
        Parameters:
        bounds: dictionary with parameter bounds {param: (min, max)}
        pop_size: population size
        n_generations: number of generations
        crossover_prob: crossover probability
        mutation_prob: mutation probability
        """
        self.bounds = bounds
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
        # Parameter names
        self.param_names = list(bounds.keys())
        self.n_params = len(self.param_names)
        
        # Setup DEAP
        self._setup_deap()
        
        # Simulation parameters
        self.sim_time = 15.0
        self.dt = 0.01
        
    def _setup_deap(self):
        """Setup DEAP framework for genetic algorithm"""
        # Clear any existing classes to avoid warnings
        try:
            del creator.FitnessMin
            del creator.Individual
        except:
            pass
        
        # Create fitness and individual classes
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # Create toolbox
        self.toolbox = base.Toolbox()
        
        # Store bounds for use in operations
        param_bounds = self.bounds
        
        # Define attribute generator for each parameter
        for param_name, (min_val, max_val) in param_bounds.items():
            self.toolbox.register(f"attr_{param_name}", 
                                 np.random.uniform, min_val, max_val)
        
        # Create individual from attributes
        attr_funcs = [getattr(self.toolbox, f"attr_{param}") 
                     for param in self.param_names]
        
        self.toolbox.register("individual", tools.initCycle, 
                             creator.Individual, attr_funcs, n=1)
        
        # Create population
        self.toolbox.register("population", tools.initRepeat, 
                             list, self.toolbox.individual)
        
        # Get bounds for mutation and crossover
        lows = [param_bounds[p][0] for p in self.param_names]
        ups = [param_bounds[p][1] for p in self.param_names]
        
        # Register genetic operators
        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                             low=lows, up=ups, eta=20.0)
        
        self.toolbox.register("mutate", tools.mutPolynomialBounded,
                             low=lows, up=ups, eta=20.0, indpb=0.2)
        
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Register evaluation function
        self.toolbox.register("evaluate", self.evaluate_individual)
        
    def evaluate_individual(self, individual):
        """
        Evaluate fitness of an individual (set of parameters)
        
        Parameters:
        individual: list of parameter values
        
        Returns:
        tuple: fitness value (lower is better)
        """
        # Convert individual to parameter dictionary
        params = {}
        for i, name in enumerate(self.param_names):
            params[name] = individual[i]
        
        # Run simulation
        try:
            metrics = self._run_simulation(params)
            
            # Combine metrics into single fitness value
            # Lower is better
            fitness = (
                0.4 * metrics['ITAE'] +      # Tracking error (primary)
                0.3 * metrics['ISE'] +       # Error energy
                0.2 * metrics['max_control'] +  # Control effort
                0.1 * metrics['steady_state_error']  # Steady-state error
            )
            
            # Add penalty for instability
            if metrics['max_control'] > 200:
                fitness += 1000
                
            if metrics['settling_time'] > 10:
                fitness += 500
                
            # Penalize large steady-state error
            if metrics['steady_state_error'] > 1.0:
                fitness += 1000
                
            return (fitness,)
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return (1e6,)
    
    def _run_simulation(self, params):
        """
        Run simulation with given controller parameters
        
        Parameters:
        params: controller parameters dictionary
        
        Returns:
        metrics: performance metrics dictionary
        """
        # Initialize components
        rov = ROVDynamics()
        current = OceanCurrent(current_speed=0.2, direction=np.pi/4)
        
        # Create controller with given parameters
        controller = L1AdaptiveController(**params)
        
        # Simulation time
        t = np.arange(0, self.sim_time, self.dt)
        
        # Storage
        states = []
        controls = []
        
        # Reference trajectory (smooth step)
        def reference(t):
            if t < 2.0:
                return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                tau = 0.5
                t_adj = t - 2.0
                x_ref = 5.0 * (1 - np.exp(-t_adj / tau))
                return np.array([x_ref, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Run simulation
        for t_step in t:
            ref = reference(t_step)
            state = rov.get_state()
            pos = state[:6]
            vel = state[6:]
            current_vel = current.get_current_velocity(pos[:3], t_step)
            
            control, _, _ = controller.compute_control(ref, pos, vel, current_vel, t_step)
            rov.apply_control(control)
            rov.integrate(self.dt, current_vel[:6])
            
            states.append(state.copy())
            controls.append(control.copy())
        
        # Convert to arrays
        states = np.array(states)
        controls = np.array(controls)
        
        # Calculate metrics
        ref_traj = np.array([reference(t_step) for t_step in t[:len(states)]])
        pos_error = states[:, :6] - ref_traj[:, :6]
        
        from scipy.integrate import trapezoid
        
        metrics = {
            'ISE': trapezoid(np.sum(pos_error**2, axis=1), t[:len(states)]),
            'IAE': trapezoid(np.sum(np.abs(pos_error), axis=1), t[:len(states)]),
            'ITAE': trapezoid(t[:len(states)] * np.sum(np.abs(pos_error), axis=1), t[:len(states)]),
            'max_control': np.max(np.abs(controls)),
            'settling_time': self._settling_time(pos_error[:, 0], t[:len(states)]),
            'steady_state_error': np.abs(pos_error[-100:, 0]).mean() if len(pos_error) > 100 else np.abs(pos_error[:, 0]).mean()
        }
        
        return metrics
    
    def _settling_time(self, error, t, threshold=0.05):
        """Calculate settling time"""
        # Find when error stays within threshold
        within_threshold = np.abs(error) < threshold
        
        # Find first time when error enters and stays
        if np.any(within_threshold):
            # Find first index where error is within threshold
            first_entry = np.argmax(within_threshold)
            
            # Check if it stays within threshold after first entry
            if np.all(within_threshold[first_entry:]):
                return t[first_entry]
            
        return t[-1]
    
    def optimize(self):
        """
        Run genetic algorithm optimization
        
        Returns:
        best_params: dictionary of best parameters
        best_fitness: best fitness value
        history: list of best fitness per generation
        """
        print("Initializing GA optimization...")
        print(f"Parameter space: {self.param_names}")
        print(f"Population size: {self.pop_size}")
        print(f"Generations: {self.n_generations}")
        
        # Create initial population
        pop = self.toolbox.population(n=self.pop_size)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        
        # Hall of Fame to store best individual
        hof = tools.HallOfFame(1)
        
        # ==========================================
        # PERBAIKAN: Buat list untuk menyimpan history
        # ==========================================
        history_records = []
        
        # Run algorithm with verbose output
        print("\nStarting evolution...")
        for gen in range(self.n_generations):
            # Evaluate individuals
            fitnesses = list(map(self.toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
            
            # Select next generation
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if np.random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            pop[:] = offspring
            
            # Update statistics
            record = stats.compile(pop)
            
            # ==========================================
            # PERBAIKAN: Simpan record ke dalam list
            # ==========================================
            history_records.append(record)
            
            print(f"Generation {gen+1}/{self.n_generations} - Min: {record['min']:.4f}, Avg: {record['avg']:.4f}")
        
        # Get best individual
        best_individual = tools.selBest(pop, 1)[0]
        best_params = {}
        for i, name in enumerate(self.param_names):
            best_params[name] = best_individual[i]
        
        best_fitness = best_individual.fitness.values[0]
        
        # ==========================================
        # PERBAIKAN: Ekstrak data dari history_records
        # ==========================================
        history = {
            'min': [record['min'] for record in history_records],
            'avg': [record['avg'] for record in history_records],
            'std': [record['std'] for record in history_records]
        }
        
        return best_params, best_fitness, history


def tune_controller_parameters(bounds, pop_size=50, n_generations=30):
    """
    Convenience function to tune controller parameters
    
    Parameters:
    bounds: parameter bounds dictionary
    pop_size: population size
    n_generations: number of generations
    
    Returns:
    best_params: optimized parameters
    """
    tuner = GATuner(bounds, pop_size, n_generations)
    best_params, best_fitness, history = tuner.optimize()
    return best_params


if __name__ == "__main__":
    # Example usage
    bounds = {
        'omega_c': (5.0, 50.0),
        'gamma': (100.0, 1000.0),
        'k': (5.0, 30.0),
        'A_m': (-50.0, -5.0),
        'B_m': (5.0, 50.0)
    }
    
    best_params = tune_controller_parameters(bounds, pop_size=20, n_generations=5)
    print("\nOptimized Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value:.4f}")