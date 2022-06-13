import ioh
import numpy as np

from .algorithm import Algorithm

class SimulatedAnnealing(Algorithm):
    def __init__(self, kmax = 1_000):
        self.kmax = kmax

    def __call__(self, problem: ioh.problem.Real):
        pass