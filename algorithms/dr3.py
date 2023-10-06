from dataclasses import dataclass

import numpy as np
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET

@dataclass
class DR3(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET    
    lambda_: int = 10

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        dim = problem.meta_data.n_variables
        beta = np.sqrt(1 / dim)
        c = beta 
        zeta = np.array([2/3, 3/2])
        m = (3 / 2) * (dim**2)
        cm = (1 / np.sqrt(m)) * (1 + (1 / m))
        cu = np.sqrt((2 - c) / c)
        m = np.ceil(1.5 * dim**2).astype(int)

        B = (1 / dim) * np.random.normal(size=(m, dim))
        B[0, :] = 0 
        sigma = 1.
        x_prime = np.random.uniform(problem.bounds.lb, problem.bounds.ub)

        while self.not_terminate(problem, self.lambda_):
            z = np.random.normal(size=(m, self.lambda_))
            zeta_i = np.random.choice(zeta, self.lambda_)
            y = cm * B.T.dot(z)
            x = x_prime + (sigma * zeta_i * y).T
            f = np.array([problem(xi) for xi in x])
            
            idx = np.argmin(f)
            x_prime = x[idx].copy()
            
            sigma *= np.power(zeta_i[idx], beta)
            b = ((1 - c) * B[0, :]) + (c * (cu * zeta_i[idx] * y[:, idx]))
            B = np.vstack([b, B])[:-1]
