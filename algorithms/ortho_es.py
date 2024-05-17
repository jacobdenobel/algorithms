import itertools
from dataclasses import dataclass
from scipy.stats import ortho_group
from scipy.linalg import qr
import matplotlib.pyplot as plt

import numpy as np
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET
from .utils import Weights, init_lambda


@dataclass
class OrthogonalES(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    lambda_: int = 4
    mu: float = None
    sigma0: float = .5
    verbose: bool = True

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        n = problem.meta_data.n_variables  
        
        Z = np.array(list(itertools.product(*zip(-np.ones(n), np.ones(n))))).T
        
        
        # H = np.random.rand(n, self.lambda_ // 2)
        # u, s, vh = np.linalg.svd(H, full_matrices=False)
        # Z = u @ vh
        # Z = np.c_[Z, -Z]
        
        self.lambda_ = len(Z.T)
        self.mu = self.mu or self.lambda_ // 2
       

        weights = Weights(self.mu, self.lambda_, n)
        x_prime = np.zeros((n, 1))
        sigma = self.sigma0
        s = np.ones((n, 1))
        
        try:
            while not self.should_terminate(problem, self.lambda_):
                # Z = Z_total.copy()
                X = x_prime + (sigma * Z)
                f = problem(X.T) 
                idx = np.argsort(f)

                mu_best = idx[: self.mu]
                
                z_prime = np.sum(weights.w * Z[:, mu_best], axis=1, keepdims=True)
                x_prime = x_prime + (sigma * z_prime)
                
                s = ((1 - weights.c_s) * s) + (weights.sqrt_s * z_prime)
                sigma = sigma * np.exp(weights.c_s / weights.d_s * (np.linalg.norm(s) / np.linalg.norm(Z) - 1))
                
        except KeyboardInterrupt:
            pass
        return x_prime
