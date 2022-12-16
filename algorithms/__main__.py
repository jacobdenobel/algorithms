from .particle_swarm_optimization import ParticleSwarmOptimization
from .plant_propagation_algorithm import PlantPropagationAlgorithm
from .genetic_algorithm import GeneticAlgorithm
from .simulated_annealing import SimulatedAnnealing
from .differential_evolution import DifferentialEvolution
from .evolution_strategy import EvolutionStrategy
from .one_plus_one_es import OnePlusOneES
from .unbounded_integer_ea import UnboundedIntegerEA, test_discrete_bbob
from .gsemo import GSEMO, gsemo_onemax_zeromax
from .dr1 import DR1
from .dr2 import DR2
from .dr3 import DR3


real = (
    ParticleSwarmOptimization, 
    SimulatedAnnealing,
    EvolutionStrategy,
    OnePlusOneES,
    DifferentialEvolution,
)

binary = (
    GeneticAlgorithm,
)

other = (
    PlantPropagationAlgorithm,
    UnboundedIntegerEA, 
)

multi = (
    GSEMO, 
)

if __name__ == '__main__':
    import ioh
    import numpy as np
    test_discrete_bbob(dim=100, budget=100_000, stepsize=.01)
    # d = 10
    # np.random.seed(10)
    # p = ioh.get_problem(1, 1, d)
    # alg = DifferentialEvolution(np=10*d)
    # alg(p)
    # print(p.state)
    