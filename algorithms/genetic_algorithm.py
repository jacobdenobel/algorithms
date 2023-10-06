from dataclasses import dataclass

import ioh
import numpy as np

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET

@dataclass
class GeneticAlgorithm(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    mu: int = 2
    lambda_: int = 100
    mu_plus_lambda: bool = False
    pm: float = None
    verbose: bool = False
    int_as_binary: bool = True
    maximize: bool = False
    use_p: bool = True
    uniform_crossover: bool = True
   
    def n_variables_binary(self, problem):
        delta = problem.bounds.ub[0] - problem.bounds.lb[0]
        return (len(bin(delta)) - 2)
    
    def decode(self, x):
        as_int = np.sum(self.powers2 * np.vstack(np.split(x, self.nint)), axis=1)
        as_int = (as_int - self.ub).clip(-self.ub, self.ub)
        return as_int


    def __call__(self, problem: ioh.problem.IntegerSingleObjective) -> SolutionType:
        dim = problem.meta_data.n_variables

        fitness_func = problem
        self.ub = problem.bounds.ub[0]
        self.nint = dim

        p = np.ones(dim) / dim

        is_binary = problem.bounds.lb[0] == 0 and problem.bounds.ub[0] == 1

        if self.int_as_binary and not is_binary:
            self.nbin = self.n_variables_binary(problem) 
            self.powers2 =  (2 ** np.arange(self.nbin))[::-1]
            dim = self.nbin * self.nint
            
            if self.use_p:
                p = np.tile(self.powers2[::-1], self.nint)
                p = p / p.sum()
            else:    
                p = np.ones(dim) / dim
            fitness_func = lambda x: problem(self.decode(x))
            is_binary = True

        pm = self.pm or (1 / dim)
        pc = .9
        # Initialize population
        parents = np.random.choice([0, 1], size=(self.mu, dim))
        fitness = np.array([fitness_func(x) for x in parents])

        offspring = np.empty((self.lambda_, dim), dtype=int)
        offspring_fitness = np.empty(self.lambda_)
        selected = []
        while self.not_terminate(problem, self.lambda_):
            # Select mu parents (Rank selection)
            idx = np.argsort(fitness)

            if self.maximize:
                idx = idx[::-1]

            idx = idx[: self.mu]
            fitness = fitness[idx]
            parents = parents[idx, :]

            # Recombine lambda offspring (1-point crossover)
            pidx = np.random.choice(range(self.mu), size=self.lambda_ * 2)
            cidx = np.random.choice(range(0, dim - 1), size=self.lambda_)
            
            for i, (c, p1, p2) in enumerate(zip(cidx, pidx[::2], pidx[1::2])):
                
                if self.uniform_crossover:
                    mask = np.random.randint(0, 2, size=dim)
                    offspring[i] = (parents[p1] * mask) + (parents[p2] * np.abs(1 - mask))
                else:
                    offspring[i] = np.r_[parents[p1, :c], parents[p2, c:]].copy()
              

                # Mutate offspring (bit-flip mutation)
                n = max(np.random.binomial(dim, pm), 1)
                idx = np.random.choice(dim, n, False, p=p)
                if is_binary:
                    offspring[i, idx] = np.abs(1 - offspring[i, idx])
                else:
                    offspring[i, idx] = np.random.randint(problem.bounds.lb[0], problem.bounds.ub[1]+1, size=len(idx))

                # idx = np.where(np.random.uniform(size=dim) < p)[0]
                # offspring[i, idx] = np.abs(1 - offspring[i, idx])
                # selected.extend(np.where(idx)[0])

                # Compute fitness
                offspring_fitness[i] = fitness_func(offspring[i])

            if self.mu_plus_lambda:
                parents = np.vstack([parents, offspring])
                fitness = np.r_[fitness, offspring_fitness]
            else:
                parents = offspring
                fitness = offspring_fitness

            if problem.state.optimum_found:
                break
        # import collections
        # print(sorted(collections.Counter(selected).items()))
        # breakpoint()
        return problem.state.current_best.y, problem.state.current_best.x



def binom(x, dim, imax, shiftmax=10, pm=0.1):
    '''For integer'''

    pm = pm or 1. / dim
    idx = np.random.choice(dim, max(np.random.binomial(dim, pm), 1))
    xn = x.copy()
    xn[idx] += (np.random.binomial(imax, shiftmax / imax, size=idx.size) 
        * np.random.choice([-1, 1],size=idx.size)).astype(int)
    return np.clip(xn, 0, imax)