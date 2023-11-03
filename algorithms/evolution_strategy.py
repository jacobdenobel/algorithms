from dataclasses import dataclass, field
import numpy as np
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET

@dataclass
class Individual:
    x: np.array
    sigma: float
    n: int
    f:float = np.inf    

    @staticmethod
    def create(lb, ub, fn, n):
        # Design choice for sigma0
        sigma = np.random.uniform(0, np.sqrt(np.abs(ub - lb)))
        x = np.random.uniform(lb, ub, size=n)
        return Individual(x, sigma, n, fn(x))

    @staticmethod
    def create_mu(mu, *args, **kwargs):
        return [Individual.create(*args, **kwargs) 
                for _ in range(mu)]

    def recombine(self, other):
        """Uniform recombination"""

        I = np.ones(self.x.size)
        U = np.random.randint(0, 2, size=self.x.size)
        return Individual(
            (U * self.x) + ((I -U) * other.x),
            (self.sigma + other.sigma) / 2,
            self.n
        )

    def mutate(self, tau2):
        self.sigma *= np.random.lognormal(0, tau2)
        self.x += self.sigma * np.random.normal(size=self.n)


@dataclass
class EvolutionStrategy(Algorithm):
    """Simple ES"""
    budget: int = DEFAULT_MAX_BUDGET
    mu: int = 4
    lamb: int = 28
    plus: bool = True

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        dim = problem.meta_data.n_variables

        tau = 1 / np.sqrt(dim)
        tau2 = pow(tau, 2)
               
        population = Individual.create_mu(self.mu, problem.bounds.lb[0], problem.bounds.ub[0], problem, dim)
        
        while not self.should_terminate(problem, self.lamb):
            offspring = []

            for _ in range(self.lamb):
                p1, p2 = np.random.choice(population, 2, replace=False)
                c = p1.recombine(p2)
                c.mutate(tau2)
                c.f = problem(c.x)
                offspring.append(c)
            
            population = sorted(
                population + offspring if self.plus else offspring,
                key=lambda i:i.f
            )[:self.mu]

        return (population[0].f, population[0].x,)


