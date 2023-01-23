from dataclasses import dataclass

import numpy as np
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET

@dataclass
class DR2(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET    
    lambda_: int = 10
    sigma0: float = 1
    greedy: bool = False

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        dim = problem.meta_data.n_variables
        beta_scale = 1 / dim
        beta = np.sqrt(beta_scale)
        c =  beta

        zeta = np.zeros((dim, 1))
        sigma_local = np.ones((dim, 1)) * self.sigma0
        sigma = self.sigma0

        c1 = np.sqrt(c / (2 - c))
        c2 = np.sqrt(dim) * c1

        x_prime = np.random.uniform(problem.bounds.lb, problem.bounds.ub)
        fbest = float("inf")
        while problem.state.evaluations < self.budget and not problem.state.optimum_found:
            z = np.random.normal(size=(dim, self.lambda_))

            x = x_prime + (sigma * (sigma_local * z)).T
            
            x = x.clip(problem.bounds.lb, problem.bounds.ub)
            f = np.array([problem(xi) for xi in x])
            
            idx = np.argmin(f)
            if self.greedy and f[idx] < fbest:
                fbest = f[idx]
                x_prime = x[idx, :].copy()
            elif not self.greedy:
                x_prime = x[idx, :].copy()

            zeta = ((1 - c) * zeta) + (c * z[:, idx].reshape(-1, 1))
            sigma *= np.power(np.exp((np.linalg.norm(zeta) / c2) - 1 + (1 / (5*dim))), beta)
            sigma_local *= np.power((np.abs(zeta) / c1) + (7 / 20), beta_scale)         
