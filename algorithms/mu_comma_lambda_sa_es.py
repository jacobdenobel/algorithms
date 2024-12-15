"""See: Evolution Strategies - Nikolaus Hansen, Dirk V. Arnold and Anne Auger (2015) Algorithm 2"""

from dataclasses import dataclass, field
import numpy as np
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET
from .sampling import Sampler, Normal



@dataclass
class SaEvolutionStrategy(Algorithm):
    """Simple ES"""
    budget: int = DEFAULT_MAX_BUDGET
    mu: int = None
    lamb: int = None
    sampler: Sampler = Normal()
    sigma_sampler: Sampler = Normal()

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        n = problem.meta_data.n_variables
        self.lamb = self.lamb or 5 * n 
        self.mu = self.mu or int(np.ceil(self.lamb / 4))

        tau = 1 / np.sqrt(n)
        tau_i = 1 / pow(n, 0.25)
        
        m = np.random.uniform(problem.bounds.lb, problem.bounds.ub)
        sigma = pow((problem.bounds.ub - problem.bounds.lb), .25)
            
        while not self.should_terminate(problem, self.lamb):
            global_dsigma = np.exp(tau * self.sigma_sampler(self.lamb)).reshape(-1, 1)
            dsigma = np.exp(tau_i * self.sigma_sampler.sample_k(n, self.lamb))
            sigma_prime = sigma * global_dsigma * dsigma
            
            Z = self.sampler.sample_k(n, self.lamb)
            X = m + (sigma_prime * Z)
            f = np.array(problem(X))
            
            # select          
            mu_best = np.argsort(f)[:self.mu]
        
            # adapt
            sigma = np.mean(sigma_prime[mu_best], axis=0)
            m = np.mean(X[mu_best], axis=0)
        
            
