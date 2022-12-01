from dataclasses import dataclass

import numpy as np
from scipy.stats import skew
from tqdm import trange
import ioh


@dataclass
class Individual:
    sigma: float
    x: np.ndarray
    y: float = float("inf") 
    
    def mutate(self):
        # Mutate sigma by a lognormal dist. of zero mean and 1/n variance.
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
class UnboundedIntegerEA:
    mu: int
    lambda_: int
    n_iterations: int = 1000
    current: int = 0
    sigma0: float = None
    verbose: bool = False

    def __call__(self, problem: ioh.problem.Integer):
        self.current = 0
        
        # Sigma proportional to the nth root of the starting area M
        self.sigma0 = self.sigma0 or np.prod(
            pow(
                np.abs((problem.bounds.lb - problem.bounds.ub)),  
                1 / problem.meta_data.n_variables
            )
        )

        # Initialize parent population
        pop = [
            Individual(
                sigma=np.random.uniform(0, 1) * self.sigma0, 
                x=(xi := np.random.randint(problem.bounds.lb, problem.bounds.ub)),
                y=problem(xi)
            ) for _ in range(self.mu)
        ]
        
        while not self.done and not problem.state.optimum_found:
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

            if self.current % 10 == 0 and self.verbose:
                print(self.current, problem.state)
            

    @property
    def done(self):
        if self.current >= self.n_iterations:
            return True
        self.current += 1

def f1(x: np.ndarray):
    return np.linalg.norm(x, ord = 1)

def ca(dim, _):
    return [0] * dim, 0.0


if __name__ == "__main__":
    np.random.seed(10)
    n_trails = 100
    n_iterations = 2000

    p1 = ioh.wrap_problem(f1, "f1", "Integer", 30, lb=-1000, ub=1000, calculate_objective=ca)
    ea = UnboundedIntegerEA(30, 100, n_iterations=n_iterations)
    
    n_gens = []
    for i in trange(n_trails):
        ea(p1)
        n_gens.append(ea.current)
        p1.reset()
    
    print(
        f"Min: {np.min(n_gens)}, Max: {np.max(n_gens)}, "
        f"Mean: {np.mean(n_gens)}, std.dev: {np.std(n_gens)}, "
        f"skew: {skew(n_gens)}"
    )

