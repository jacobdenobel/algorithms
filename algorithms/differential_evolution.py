from dataclasses import dataclass

import ioh
import numpy as np

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET

@dataclass
class DifferentialEvolution(Algorithm):
    budget:int = DEFAULT_MAX_BUDGET
    np: int = 10
    cr: float = .9
    f: float = .8

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        population = np.random.uniform(
            problem.bounds.lb,
            problem.bounds.ub,
            (self.np, problem.meta_data.n_variables),
        )
        f = np.array([problem(x) for x in population])

        pmask = [np.where(x)[0] for x in np.abs(np.eye(self.np) - 1).astype(bool)]

        while problem.state.evaluations <= (self.budget - self.np) and not problem.state.optimum_found:
            for i, m in enumerate(pmask):
                x, y = population[i].copy(), f[i]
                a, b, c = population[np.random.choice(m, size=3, replace=False)]

                idx = (
                    np.random.uniform(0, 1, size=problem.meta_data.n_variables)
                    < self.cr
                )
                idx[np.random.choice(problem.meta_data.n_variables)] = True

                x[idx] = a[idx] + (self.f * (b[idx] - c[idx]))
                if (y_ := problem(x)) < y:
                    population[i] = x
                    f[i] = y_

                if problem.state.optimum_found:
                    break

        idxmin = np.argmin(f)
        return f[idxmin], population[idxmin]
