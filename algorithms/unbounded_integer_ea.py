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
        rs = np.random.normal(0, np.sqrt(1 / self.x.size))
        self.sigma = max(self.sigma * np.exp(rs), 1.0)
        
        # Equation (7)
        s_over_n = self.sigma / self.x.size
        p = 1 - (s_over_n / (np.sqrt(1 + pow(s_over_n, 2)) + 1))


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
    # This does not work
    n_sigma: bool = False
    verbose: bool = False

    def __call__(self, problem: ioh.problem.IntegerSingleObjective):
        # Sigma proportional to the nth root of the starting area M
        self.sigma0 = self.sigma0 or np.prod(
            pow(
                np.abs((problem.bounds.lb - problem.bounds.ub)),  
                1 / problem.meta_data.n_variables
            )
        )

        if self.n_sigma:
            self.sigma0 *= np.ones(problem.meta_data.n_variables) 

        # Initialize parent population
        pop = [
            Individual(
                sigma=np.random.uniform(0, 1, size=self.sigma0.size) * self.sigma0, 
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

            if ((problem.state.evaluations - self.mu) / self.lambda_ ) % 10 == 0 and self.verbose:
                print(problem.state.evaluations)
                print(np.array([x.sigma for x in pop]).mean(axis=0))
                print(problem.state.current_best.x - problem.function.optimum.x)
                print()

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
    function: ioh.problem.RealSingleObjective
    step: float = 0.1
    as_integer: bool = True

    def __post_init__(self):
        self.translation = self.function.optimum.x % self.step
        self.bound_underflow = (self.bounds.lb * self.step) + self.translation
        self.bound_overflow = (self.bounds.ub * self.step) + self.translation
    
    def translate(self, x):
        x_prime = np.atleast_1d(x).astype(float)
        if self.as_integer:
            x_prime = np.round(x).astype(float)
            x_prime *= self.step
            
        else:
            x_prime = x - (x % self.step)
        
        x_prime = x_prime + self.translation
        x_prime[(x_prime < self.function.bounds.lb) & (x_prime >= self.bound_underflow)] = self.function.bounds.lb[0]
        x_prime[(x_prime > self.function.bounds.ub) & (x_prime <= self.bound_overflow)] = self.function.bounds.ub[0]

        return x_prime
    
    def __call__(self, x):
        x_prime = self.translate(x)
        return self.function(x_prime)
        
    @property
    def meta_data(self):
        return self.function.meta_data

    @property
    def state(self):
        return self.function.state

    
    def get_bounds(self, as_integer):
        bounds = self.function.bounds
        if not as_integer:
            return Bounds(bounds.lb, bounds.ub)
        
        return Bounds(
            np.floor(bounds.lb / self.step).astype(int),
            np.floor(bounds.ub / self.step).astype(int),
        )
    
    @property
    def bounds(self):
        return self.get_bounds(self.as_integer)
    
    @property
    def n_values(self):
        bounds = self.get_bounds(True)
        return (bounds.ub[0] - bounds.lb[0]) + 1
        

def test_discrete_bbob(pid=1, iid=1, dim=100, stepsize=.1, budget=1e4):
    np.random.seed(20)
    bbob_function = ioh.get_problem(pid, iid, dim)
    problem = DiscreteBBOB(bbob_function, step=stepsize)
    ea = UnboundedIntegerEA(4, 28, budget=budget)
    ea(problem)

    print(problem.state) 
    # print(problem.function.optimum)
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

