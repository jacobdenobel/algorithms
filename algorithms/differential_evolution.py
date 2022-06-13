import ioh
import numpy as np

from .algorithm import Algorithm, SolutionType


class DifferentialEvolution(Algorithm):
    def __init__(self, np=10, cr=0.9, f=.5, max_iterations=1000):
        self.np = np
        self.cr = cr
        self.f = f
        self.max_iterations = max_iterations

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        population = np.random.uniform(
            problem.constraint.lb,
            problem.constraint.ub,
            (self.np, problem.meta_data.n_variables),
        )
        f = np.array([problem(x) for x in population])

        pmask = [np.where(x)[0] for x in np.abs(np.eye(self.np) - 1).astype(bool)]

        for _ in range(self.max_iterations):
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
        print(problem.state)
        return f[idxmin], population[idxmin]
