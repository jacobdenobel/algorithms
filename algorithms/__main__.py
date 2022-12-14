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

    exp = ioh.Experiment(
        DR1(),
        range(1, 25),
        [1],
        [5],
        reps=10,
        logged=True,
        zip_output=True,
        remove_data=True                
    )
    exp.run()

    
