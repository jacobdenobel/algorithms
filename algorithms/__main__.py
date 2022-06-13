from .particle_swarm_optimization import ParticleSwarmOptimization
from .plant_propagation_algorithm import PlantPropagationAlgorithm
from .genetic_algorithm import GeneticAlgorithm
from .simulated_annealing import SimulatedAnnealing
from .differential_evolution import DifferentialEvolution

if __name__ == '__main__':
    import ioh
    problem = ioh.problem.Sphere(1, 2)
    print(DifferentialEvolution()(problem))