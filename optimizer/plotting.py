import numpy as np
from deap import base, creator, tools, algorithms
from plant.rov_dynamics import ROVPlant, ocean_current_disturbance
from controller.l1_adaptive import L1AdaptiveController

# Fitness function: Minimize Integral Square Error (ISE)
def evaluate_rov(individual):
    Gamma, omega = individual
    if Gamma <= 0 or omega <= 0:
        return 1e6, # Penalti jika nilai negatif
        
    dt = 0.01
    t_end = 20.0
    steps = int(t_end / dt)
    
    # Inisialisasi Plant dan Controller
    plant = ROVPlant(m=50.0, d=20.0) # Parameter ROV hipotetik
    
    # Matriks referensi (Desired Dynamics)
    wn = 1.5; zeta = 0.8 # Frekuensi natural dan damping ratio target
    Am = np.array([[0, 1], [-wn**2, -2*zeta*wn]])
    Bm = np.array([[0], [wn**2]])
    
    controller = L1AdaptiveController(Am, Bm, Gamma, omega)
    
    ise = 0.0
    setpoint = 10.0 # Target posisi maju 10 meter
    
    for i in range(steps):
        t = i * dt
        dist = ocean_current_disturbance(t)
        
        u = controller.compute_control(plant.state, setpoint, dt)
        pos = plant.update(u, dist, dt)
        
        error = setpoint - pos
        ise += (error**2) * dt
        
    return ise,

def run_ga_tuning():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    # Range tuning: Gamma [10, 10000], omega [1, 100]
    toolbox.register("attr_gamma", np.random.uniform, 10, 5000)
    toolbox.register("attr_omega", np.random.uniform, 1, 50)
    toolbox.register("individual", tools.initCycle, creator.Individual, 
                     (toolbox.attr_gamma, toolbox.attr_omega), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_rov)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=20)
    hof = tools.HallOfFame(1)
    
    print("Memulai proses Auto-Tuning GA...")
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=15, 
                        halloffame=hof, verbose=False)
    
    best_gamma, best_omega = hof[0]
    print(f"Tuning Selesai! Parameter Optimal:\nGamma = {best_gamma:.2f}, Omega = {best_omega:.2f}")
    return best_gamma, best_omega