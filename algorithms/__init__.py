from .particle_swarm_optimization import ParticleSwarmOptimization
from .plant_propagation_algorithm import PlantPropagationAlgorithm
from .genetic_algorithm import GeneticAlgorithm
from .simulated_annealing import SimulatedAnnealing
from .differential_evolution import DifferentialEvolution
from .evolution_strategy import EvolutionStrategy
from .one_plus_one_es import OnePlusOneES
from .unbounded_integer_ea import UnboundedIntegerEA, test_discrete_bbob, DiscreteBBOB
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


def ert(states):
    n_suc = 0
    total = 0
    for state in states:
        n_suc += int(state.optimum_found)
        total += state.evaluations
    return float("inf") if n_suc == 0 else total / n_suc



if __name__ == '__main__':
    import ioh
    import numpy as np
    
    fid = 1
    dim = 5
    budget = 50_000
    stepsize = .01
    reps = 10
    np.random.seed(10)

    ga = GeneticAlgorithm(budget, mu=4, lambda_=28, mu_plus_lambda=True, int_as_binary=False)
    states = []
    for i in range(reps):
        p = ioh.get_problem(1, 1, dim)
        dp = DiscreteBBOB(p, stepsize, as_integer=True)
        ga(dp)
        states.append(dp.state)

    print(ert(states))