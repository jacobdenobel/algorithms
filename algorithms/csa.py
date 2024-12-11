from dataclasses import dataclass

import numpy as np
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET
from .utils import Weights, init_lambda
from .sampling import Sampler, Normal


@dataclass
class CSA(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    lambda_: int = None
    mu: float = None
    sigma0: float = .5
    verbose: bool = True
    mirrored: bool = False
    sampler: Sampler = Normal()

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        n = problem.meta_data.n_variables  
        self.lambda_ = self.lambda_ or init_lambda(n, "default")
        self.mu = self.mu or self.lambda_ // 2

        weights = Weights(self.mu, self.lambda_, n)

        echi = self.sampler.expected_length(n)
        x_prime = np.zeros((n, 1))
        self.sigma = self.sigma0

        s = np.zeros((n, 1))
        n_samples = self.lambda_ if not self.mirrored else self.lambda_ // 2

        try:
            while not self.should_terminate(problem, self.lambda_):
                Z = self.sampler.sample_k(n, n_samples).T
                if self.mirrored:
                    Z = np.hstack([Z, -Z])
                X = x_prime + (self.sigma * Z)
               
                f = problem(X.T) 
                idx = np.argsort(f)

                mu_best = idx[: self.mu]
                z_prime = np.sum(weights.w * Z[:, mu_best], axis=1, keepdims=True)
                x_prime = x_prime + (self.sigma * z_prime)
                s = ((1 - weights.c_s) * s) + (weights.sqrt_s * z_prime)

                self.sigma = self.sigma * np.exp(weights.c_s / weights.d_s * (np.linalg.norm(s) / echi - 1))

                # print(problem.state.evaluations, sigma, np.mean(f), problem.state.current_best.y)
                
        except KeyboardInterrupt:
            pass
        return x_prime


