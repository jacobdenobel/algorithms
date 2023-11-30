import time
from dataclasses import dataclass

import numpy as np
import ioh

from ..algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET

@dataclass
class SalimansES(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    lr: float = 0.2
    sigma: float = 0.01
    lambda_: int = 10

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        n = problem.meta_data.n_variables
        x_prime = np.zeros(n)
        f_prime = problem(x_prime)
        try:
            while not self.should_terminate(problem, self.lambda_ * 2):
                noise = np.random.normal(size=(self.lambda_, n))
                noise *= self.sigma / np.sqrt(n)
                noise *= (np.array(problem(x_prime + noise)) - np.array(problem(x_prime - noise))).reshape(-1, 1)
                grad = noise.sum(axis=0)
                g_hat = grad / (2 * self.lambda_ * self.sigma ** 2)
                x_prime -= self.lr * g_hat
                f_prime = problem(x_prime)
        except KeyboardInterrupt:
            pass

        return x_prime, f_prime
