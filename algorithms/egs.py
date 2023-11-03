from dataclasses import dataclass

import numpy as np
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET


@dataclass
class EGS(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    sigma0: float = 0.02     
    lambda_: int = 16        
    mu: int = 1             
    kappa: float = 2.0    
    
    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        n = problem.meta_data.n_variables
        x_prime = np.zeros((n, 1))

        sigma = self.sigma0
        try:
            while not self.should_terminate(problem, self.lambda_):
                Z = np.random.normal(size=(n, self.lambda_))
                y_pos = x_prime + sigma * Z
                y_neg = x_prime - sigma * Z
                f_pos = problem(y_pos)
                f_neg = problem(y_neg)

                z_avg  = np.sum((f_neg - f_pos) * Z, axis=1, keepdims=True)
                z_prog = (np.sqrt(n) / self.kappa) * (z_avg / np.linalg.norm(z_avg))
                x_prime = x_prime + sigma * z_prog

        except KeyboardInterrupt:
            pass
        return x_prime