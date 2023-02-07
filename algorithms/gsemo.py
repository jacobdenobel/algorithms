from dataclasses import dataclass
from collections import namedtuple

import ioh
import numpy as np
import matplotlib.pyplot as plt


from .algorithm import Algorithm, DEFAULT_MAX_BUDGET

Solution = namedtuple("Solution", ["x", "y", "c"])

@dataclass
class GSEMO(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    verbose_rate: int = 0
    count_nullops: bool = False

    def __call__(self, f: ioh.problem.IntegerSingleObjective) -> None:
        def dominated(a, b):
            return a.y > b.y and a.c < b.c

        def strictly_dominates(a, b):
            return a.y >= b.y and a.c <= b.c

        def determine_new_pareto_set(p, gx):
            # Filter out dominated z from p and optionally add g(x)
            any_dominates = False
            pnew = [gx]

            for z in p:
                if strictly_dominates(z, gx):
                    any_dominates = True
                    break

                if not dominated(z, gx):
                    pnew.append(z)

            if not any_dominates:
                return pnew
            return p

        n = f.meta_data.n_variables
        x = np.random.randint(0, 2, n)
        p = [Solution(x, f(x), f.constraints[0].violation())]
        pm = (1 / n)

        for _ in range(self.budget - 1):
            # Select
            xi = p[np.random.randint(0, len(p))].x.copy()
            
            # Bitflip
            nm = max(np.random.binomial(n, pm), 1)
            idx = np.random.choice(n, nm, False)
            xi[idx] = np.logical_not(xi[idx]).astype(int)

            # Determine g(x)
            gx = Solution(xi, f(xi), f.constraints[0].violation())
            
            # Update new pareto set
            p = determine_new_pareto_set(p, gx)

            if self.verbose_rate and f.state.evaluations % self.verbose_rate == 0:
                print(sorted([(x.y, x.c) for x in p]))

        return p       

def count_zeros(x):
    return (x == 0).sum()

def get_onemax_zeromax(instance, dim):
    problem = ioh.get_problem(1, instance, dim, "Integer")
    c = ioh.IntegerConstraint(count_zeros, weight=0)
    problem.add_constraint(c)
    return problem

def get_multiobjective_submodular(fid):
    problem = ioh.get_problem(fid, 1, 1, "Integer")
    problem.constraints[0].weight = -1
    problem.constraints[0].exponent = 0.
    return problem

def plot_front(p, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    c = np.array([sol.c for sol in p])
    y = np.array([sol.y for sol in p])
    idx = np.argsort(c)
    ax.plot(c[idx], y[idx], linestyle='--', marker='o', alpha=.7)
    ax.set_xlabel("c(x)")
    ax.set_ylabel("f(x)")
    ax.grid()     

def gsemo_onemax_zeromax():
    problem = get_onemax_zeromax(1, 100)
    p = GSEMO()(problem)
    plot_front(p)
    print(problem.state)
    plt.show()


if __name__ == "__main__":
    gsemo_onemax_zeromax()