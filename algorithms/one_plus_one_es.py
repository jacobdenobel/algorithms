from dataclasses import dataclass

import numpy as np
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET

@dataclass
class OnePlusOneES(Algorithm):
    '''With 1/5 success rule'''
    budget: int = DEFAULT_MAX_BUDGET    
    sigma0: float = None

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        dim = problem.meta_data.n_variables
        a = np.random.uniform(problem.bounds.lb, problem.bounds.ub)
        f = problem(a)
        sigma = self.sigma0 or np.linalg.norm(problem.bounds.lb - problem.bounds.ub) / np.sqrt(dim)
        n_success = 0
        
        while self.not_terminate(problem):
            t = problem.state.evaluations
            if (t > 0 and t % dim == 0):
                if (n_success < (2*dim)):
                    sigma *= .85
                else:
                    sigma /= .85
                n_success = 0
            
            a0 = a + (sigma * np.random.normal(0, 1, size=dim))
            f0 = problem(a0)
            if f0 < f:
                a = a0.copy()    
                f = f0
                n_success += 1
        return (f, a,) 
        
