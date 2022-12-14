from dataclasses import dataclass

import numpy as np
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET

@dataclass
class DR1(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET    
    mu: int = 1
    lambda_: int = 10

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        dim = problem.meta_data.n_variables
        beta_scale = 1 / dim
        beta = np.sqrt(beta_scale)
        zeta = np.array([5/7, 7/5])
        sigma = np.ones((dim, 1))
        
        root_pi = np.sqrt(2/np.pi)

        x_prime = np.random.uniform(problem.bounds.lb, problem.bounds.ub)

        while problem.state.evaluations < self.budget and not problem.state.optimum_found:
            z = np.random.normal(size=(dim, self.lambda_))
            zeta_i = np.random.choice(zeta, self.lambda_)

            x = x_prime + (zeta_i * (sigma * z)).T
            f = np.array([problem(xi) for xi in x])
            
            idx = np.argmin(f)
            x_prime = x[idx, :].copy()
            
            zeta_sel = np.exp(np.abs(z[:, idx]) - root_pi)
            sigma *= (np.power(zeta_i[idx], beta) * np.power(zeta_sel, beta_scale)).reshape(-1, 1)


