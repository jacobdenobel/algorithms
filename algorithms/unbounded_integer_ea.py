'''Source: An Evolutionary Algorithm for Integer Programming - Gunter Rudolf'''
from dataclasses import dataclass

import numpy as np
from tqdm import trange
import ioh

from .algorithm import Algorithm, DEFAULT_MAX_BUDGET


@dataclass
class Individual:
    sigma: float
    x: np.ndarray
    y: float = float("inf") 
    
    def mutate(self):
        # TODO: check how we can PROPERLY add multiple N variances
        scale = self.x.size * self.sigma.size
        # Mutate sigma by a lognormal dist. of zero mean and 1/n variance.
        rs = np.random.normal(0, 1 / scale, size=self.sigma.size)
        self.sigma = np.clip(self.sigma * np.exp(rs), 1.0, np.inf)

        # Equation (7)
        s_over_n = self.sigma / scale
        p = 1 - (s_over_n / (np.sqrt(1 + np.power(s_over_n, 2)) + 1))

        # Mutate x by adding the difference of two geom. dist. random variables
        log_p = np.log(1 - p)
        g1 = np.floor(np.log(1 - np.random.uniform(0, 1, self.x.size)) / log_p).astype(int)
        g2 = np.floor(np.log(1 - np.random.uniform(0, 1, self.x.size)) / log_p).astype(int)
        self.x += (g1 - g2)

    def recombine(self, other: "Individual") -> "Individual":
        # Random uniform recombination
        mask = np.random.randint(0, 2, size=self.x.size).astype(bool)
        return Individual(
            sigma=(self.sigma + other.sigma) / 2,
            x=self.x * mask + other.x * ~mask
        )

 
@dataclass
class UnboundedIntegerEA(Algorithm):
    mu: int
    lambda_: int
    budget: int = DEFAULT_MAX_BUDGET
    sigma0: float = None
    n_sigma: bool = True 
    verbose: bool = False

    def __call__(self, problem: ioh.problem.Integer):
        # Sigma proportional to the nth root of the starting area M
        if not self.n_sigma:
            self.sigma0 = self.sigma0 or np.prod(
                pow(
                    np.abs((problem.bounds.lb - problem.bounds.ub)),  
                    1 / problem.meta_data.n_variables
                )
            )
        else:
            self.sigma0 = np.sqrt(np.abs(problem.bounds.lb - problem.bounds.ub)).astype(int)

        # Initialize parent population
        pop = [
            Individual(
                sigma=np.random.uniform(0, 1) * self.sigma0, 
                x=(xi := np.random.randint(problem.bounds.lb, problem.bounds.ub)),
                y=problem(xi)
            ) for _ in range(self.mu)
        ]
        
        # while not self.done and not problem.state.optimum_found:
        while problem.state.evaluations < (self.budget - self.lambda_) and \
                not problem.state.optimum_found:
            new_pop = []
            for _ in range(self.lambda_):
                # Randomly select two parents 
                p1, p2 = np.random.choice(pop, 2, replace=False)
                child = p1.recombine(p2)
                child.mutate()
                child.y = problem(child.x)
                new_pop.append(child)

            # Select mu best
            new_pop.sort(key=lambda i:i.y)
            pop = new_pop[:self.mu]

            if problem.state.evaluations % 100 == 0 and self.verbose:
                print(problem.state.evaluations, problem.state)

        return problem.state.current_best


def f1(x: np.ndarray):
    return np.linalg.norm(x, ord = 1)

def f2(x: np.ndarray):
    return np.linalg.norm(x, ord = 2)

def f3(x: np.ndarray):
    y = np.array([
        [35, -20, -10, 32, -10],
        [-20, 40, -6, -31, 32],
        [-10, -6, 11, -6, -10],
        [32, -31, -6, 38, -20],
        [-10, 32, -10, -20, 31],
    ])
    v = np.array([15, 27, 36, 18, 12])
    
    return -(v.dot(x) - x.T.dot(y).dot(x))

def ca(dim, _):
    return [0] * dim, 0.0

def ca2(*args):
    return np.array([0, 11, 22, 16, 6]), -737


@dataclass
class Bounds:
    lb: np.ndarray
    ub: np.ndarray

@dataclass
class DiscreteBBOB:
    function: ioh.problem.Real
    step: float = 0.1
    as_integer: bool = True

    def __post_init__(self):
        self.translation = self.function.optimum.x % self.step

    def __call__(self, x):
        x_prime = x.astype(float)
        if self.as_integer:
            x_prime *= self.step
        x_prime = x_prime + self.translation
        return self.function(x_prime)

    @property
    def meta_data(self):
        return self.function.meta_data

    @property
    def state(self):
        return self.function.state

    @property
    def bounds(self):
        bounds = self.function.bounds
        divider = self.step if self.as_integer else 1.
        t = int if self.as_integer else float
        return Bounds(
            (bounds.lb / divider).astype(t),
            (bounds.ub / divider).astype(t),
        )


def test_discrete_bbob(pid=1, iid=1, dim=100, stepsize=.1):
    np.random.seed(10)
    bbob_function = ioh.get_problem(pid, iid, dim)
    problem = DiscreteBBOB(bbob_function, step=stepsize)
    ea = UnboundedIntegerEA(10, 20, budget=dim * 1e4)
    ea(problem)
    print(problem.state) 
    return ea, problem  


def paper_experiments():
    from scipy.stats import skew
    np.random.seed(10)
    n_trails = 1000
    n_iterations = 1000
    verbose = False

    p1 = ioh.wrap_problem(f1, "f1", "Integer", 30, lb=-1000, ub=1000, calculate_objective=ca)
    p2 = ioh.wrap_problem(f2, "f2", "Integer", 30, lb=-1000, ub=1000, calculate_objective=ca)
    p3 = ioh.wrap_problem(f3, "f3", "Integer", 5, lb=0, ub=100, calculate_objective=ca2)
    ea = UnboundedIntegerEA(30, 100, n_iterations=n_iterations, verbose=verbose)
    
    for p in (p1, p2, p3,):
        n_gens = []
        for i in trange(n_trails):
            ea(p)
            n_gens.append(ea.current)
            p.reset()
        
        print(
            f"Problem: {p}: "
            f"Min: {np.min(n_gens)}, Max: {np.max(n_gens)}, "
            f"Mean: {np.mean(n_gens)}, std.dev: {np.std(n_gens)}, "
            f"skew: {skew(n_gens)}"
        )


if __name__ == "__main__":
    main()