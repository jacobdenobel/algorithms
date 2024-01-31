from dataclasses import dataclass

import numpy as np
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET
from .utils import Weights, init_lambda






@dataclass
class CSAGrad(Algorithm):
    '''Modification of Cumulative Step Size Adaptation Method, which,
    instead of standard weighted recombination, uses gradient based weights.    
    '''

    budget: int = DEFAULT_MAX_BUDGET
    lambda_: int = None
    mu: float = None
    sigma0: float = .5
    verbose: bool = True

    def normalize_weights(self, w):
        return w / np.sum(w) 

    def set_learning_rates(self, w):
        self.mueff =  1 / np.sum(np.power(w, 2))
        self.c_s = (self.mueff + 2) / (self.n + self.mueff + 5)
        self.d_s = 1 + self.c_s + 2 * max(0, np.sqrt((self.mueff - 1) / (self.n + 1)) - 1)
        self.sqrt_s = np.sqrt(self.c_s * (2 - self.c_s) * self.mueff)

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        n = problem.meta_data.n_variables  
        self.lambda_ = self.lambda_ or init_lambda(n, "default")
        self.mu = self.mu or self.lambda_ // 2

        echi = np.sqrt(n) * (1 - (1 / n / 4) - (1 / n / n / 21))
        x_prime = np.zeros((n, 1))
        sigma = self.sigma0

        s = np.ones((n, 1))
        n_samples = self.lambda_ // 2
        
        beta = 1
        alpha = .01

        try:
            while not self.should_terminate(problem, self.lambda_):
                Z = np.random.normal(size=(n, n_samples))
                Y = sigma * Z

                f_pos = np.array(problem((x_prime + Y).T))
                f_neg = np.array(problem((x_prime - Y).T))

                w = (1 / n_samples) * (f_pos - f_neg)
                breakpoint()
                # Do we need to normalize w?
                # breakpoint()

                mueff =  1 / np.sum(np.power(w, 2))
                c_s = (mueff + 2) / (n + mueff + 5)
                d_s = 1 + c_s + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1)
                sqrt_s = np.sqrt(c_s * (2 - c_s) * mueff)

                z_prime = (Z * w).sum(axis=1, keepdims=True)
                x_prime = x_prime + (sigma * z_prime)
                s = ((1 - c_s) * s) + (sqrt_s * z_prime)

                sigma = sigma * np.exp(c_s / d_s * (np.linalg.norm(s) / echi - 1))

                print(sigma, mueff)
                breakpoint()

        except KeyboardInterrupt:
            pass
        return x_prime


