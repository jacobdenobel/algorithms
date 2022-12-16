from dataclasses import dataclass

import ioh
import numpy as np

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET

@dataclass
class DifferentialEvolution(Algorithm):
    budget:int = DEFAULT_MAX_BUDGET
    np: int = 10
    f: float = .5 # [0, 2]
    cr: float = .9 # [0, 1]

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        population = np.random.uniform(
            problem.bounds.lb,
            problem.bounds.ub,
            (self.np, problem.meta_data.n_variables),
        )
        f = np.array([problem(x) for x in population])

        pmask = np.array([np.where(x)[0] for x in np.abs(np.eye(self.np) - 1).astype(bool)])
        while problem.state.evaluations <= (self.budget - self.np) and not problem.state.optimum_found:
            pmask = np.apply_along_axis(np.random.permutation, axis=1, arr=pmask)
            
            X = population.copy()
            A, B, C = (population[pmask[:, i]] for i in range(3))

            R = np.random.uniform(0, 1, size=(self.np, problem.meta_data.n_variables))
            mask = R < self.cr
            mask[:, np.random.choice(problem.meta_data.n_variables)] = True
            
            X[mask] = A[mask] + (self.f * (B[mask] - C[mask]))

            for i, x in enumerate(X):
                if (y_ := problem(x)) < f[i]:
                    population[i] = x.copy()
                    f[i] = y_


        idxmin = np.argmin(f)
        return f[idxmin], population[idxmin]

