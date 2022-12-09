from dataclasses import dataclass

import ioh
import numpy as np

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET




@dataclass
class GeneticAlgorithm(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    mu: int = 4
    lambda_: int = 20
    mu_plus_lambda: bool = True
    pm: float = None
    verbose: bool = False
   

    def __call__(self, problem: ioh.problem.Integer) -> SolutionType:
        dim = problem.meta_data.n_variables
        pm = self.pm or (1 / dim)

        # Initialize population
        parents = np.random.choice([0, 1], size=(self.mu, dim))
        fitness = np.array([problem(x) for x in parents])
        offspring = np.empty((self.lambda_, dim), dtype=int)
        offspring_fitness = np.empty(self.lambda_)

        while problem.state.evaluations <= (self.budget - self.lambda_) and not problem.state.optimum_found:
            # Select mu parents (Rank selection)
            idx = np.argsort(fitness)[::-1][: self.mu]
            fitness = fitness[idx]
            parents = parents[idx, :]

            # Recombine lambda offspring (1-point crossover)
            pidx = np.random.choice(range(self.mu), size=self.lambda_ * 2)
            cidx = np.random.choice(range(1, dim - 1), size=self.lambda_)
            
            for i, (c, p1, p2) in enumerate(zip(cidx, pidx[::2], pidx[1::2])):
                offspring[i] = np.r_[parents[p1, :c], parents[p2, c:]].copy()

                # Mutate offspring (bit-flip mutation)
                n = max(np.random.binomial(dim, pm), 1)
                idx = np.random.choice(dim, n, False)
                offspring[i, idx] = np.abs(1 - offspring[i, idx])

                # Compute fitness
                offspring_fitness[i] = problem(offspring[i])

            if self.mu_plus_lambda:
                parents = np.vstack([parents, offspring])
                fitness = np.r_[fitness, offspring_fitness]
            else:
                parents = offspring
                fitness = offspring_fitness

            if problem.state.optimum_found:
                break
        
        return problem.state.current_best.y, problem.state.current_best.x



def binom(x, dim, imax, shiftmax=10, pm=0.1):
    '''For integer'''

    pm = pm or 1. / dim
    idx = np.random.choice(dim, max(np.random.binomial(dim, pm), 1))
    xn = x.copy()
    xn[idx] += (np.random.binomial(imax, shiftmax / imax, size=idx.size) 
        * np.random.choice([-1, 1],size=idx.size)).astype(int)
    return np.clip(xn, 0, imax)