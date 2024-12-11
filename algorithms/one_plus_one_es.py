from dataclasses import dataclass
from collections import deque

import numpy as np
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET
from .sampling import Sampler, Normal

SQRT3 = np.sqrt(3)


@dataclass
class OnePlusOneES(Algorithm):
    """With 1/5 success rule"""

    budget: int = DEFAULT_MAX_BUDGET
    sigma0: float = None
    sampler: Sampler = Normal()
    archive: deque = None
    c: float = 0.817

    def step(self, problem: ioh.ProblemType):
        t = problem.state.evaluations
        a0 = self.a + (self.sigma * self.sampler(problem.meta_data.n_variables))
        f0 = problem(a0)
        if f0 < self.f:
            self.a = a0.copy()
            self.f = f0
            self.archive.append(1)
        else:
            self.archive.append(0)
        
        if t > 0 and t % problem.meta_data.n_variables == 0:
            ps = sum(self.archive) / len(self.archive)
            if ps < 1/5:
                self.sigma *= self.c
            elif ps > 1/5:
                self.sigma /= self.c
            
    def restart(self, problem: ioh.ProblemType):
        self.a = np.random.uniform(problem.bounds.lb, problem.bounds.ub)
        self.f = problem(self.a)
        self.sigma = self.sigma0 or np.linalg.norm(
            problem.bounds.lb - problem.bounds.ub
        ) / np.sqrt(problem.meta_data.n_variables)
        self.archive = deque(maxlen=10 * problem.meta_data.n_variables)

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        self.restart(problem)
        
        while not self.should_terminate(problem):
            self.step(problem)
            
        return (
            self.f,
            self.a,
        )
