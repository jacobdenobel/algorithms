import abc
from typing import Tuple

import ioh
import numpy as np

SolutionType = Tuple[float, np.ndarray]

DEFAULT_MAX_BUDGET = 10_000


class Algorithm(abc.ABC):
    budget: int = DEFAULT_MAX_BUDGET
    target: float = 1e-8

    @abc.abstractmethod
    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        pass

    def not_terminate(self, problem: ioh.ProblemType, n_evals: int = 0) -> bool:
        return (
            problem.state.evaluations + n_evals < self.budget
            and not problem.state.optimum_found
            and not ((problem.state.current_best.y - problem.optimum.y) < self.target)
        )
