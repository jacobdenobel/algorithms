from dataclasses import dataclass, field
import numpy as np
import ioh

from .es import get_t
from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET


def T(omega, n):
    '''Rotation matrix'''
    return get_t(omega, n).reshape(n, n, order='F')

@dataclass
class Individual:
    x: np.array
    sigma: np.array
    omega: np.array = field(repr=None)
    f:float = np.inf

    @staticmethod
    def create(n, lb, ub, fn):
        sigma = np.random.uniform(0, ub, n)
        x = sigma * np.random.normal(size=n)
        omega = np.random.uniform(0, 2 * np.pi, int(n * (n - 1) / 2))
        return Individual(x, sigma, omega, fn(x))

    @staticmethod
    def create_mu(mu, *args, **kwargs):
        return [Individual.create(*args, **kwargs) 
                for _ in range(mu)]

    def recombine(self, other):
        I = np.ones(self.x.size)
        U = np.random.randint(0, 2, size=self.x.size)
        return Individual(
            (U * self.x) + ((I -U) * other.x),
            (self.sigma + other.sigma) / 2,
            ((self.omega + other.omega) % (4 * np.pi)) / 2
        )

    def mutate(self, gamma, tau, eta):
        # adapt angles
        self.omega = (self.omega + (gamma * np.random.normal(size=len(self.omega)))) % (2 * np.pi)
        
        # adapt sigma
        zt = tau * np.random.normal()
        self.sigma *= np.exp(zt + (eta * np.random.normal(size=self.sigma.size)))
        
        # mutate x
        self.x += T(self.omega, self.sigma.size)\
            .dot(np.diag(self.sigma))\
            .dot(np.random.normal(size=self.sigma.size))


@dataclass
class EvolutionStrategy(Algorithm):
    """contemporary evolution strategy (TODO: check when this was contemporary)"""
    budget: int = DEFAULT_MAX_BUDGET
    mu: int = 4
    lamb: int = 8
    plus: bool = True

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        dim = problem.meta_data.n_variables
        gamma, tau, eta = 5 * np.pi / 180, (2*dim)**(-1/2), (4*dim)**(-1/4)
        population = Individual.create_mu(self.mu, dim, -1e-4, 1e-4, problem)
        
        while problem.state.evaluations < (self.budget - self.lamb) and \
            not problem.state.optimum_found:
            offspring = []
            for _ in range(self.lamb):
                p1, p2 = np.random.choice(population, 2, replace=False)
                c = p1.recombine(p2)
                c.mutate(gamma, tau, eta)
                c.f = problem(c.x)
                offspring.append(c)
            
            population = sorted(
                population + offspring if self.plus else offspring,
                key=lambda i:i.f
            )[:self.mu]

        return (population[0].f, population[0].x,)
