import abc

import numpy as np
from scipy import stats

SQRT3 = np.sqrt(3)
SQRT05 = np.sqrt(0.5)
SQRT3pi = SQRT3 / np.pi


class Sampler(abc.ABC):
    @abc.abstractmethod
    def __call__(self, dim: int) -> np.ndarray:
        pass

    def sample_k(self, n: int, k: int) -> np.ndarray:
        return np.array([self(n) for _ in range(k)])
        

    def __repr__(self) -> str:
        return self.__class__.__name__

    def expected_length(self, n: int) -> float:
        return np.sqrt(n)


class Uniform(Sampler):
    def __call__(self, dim):
        return np.random.uniform(-SQRT3, SQRT3, size=dim)


class Normal(Sampler):
    def __call__(self, dim):
        return np.random.normal(0, 1, size=dim)

    def expected_length(self, n):
        return np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * np.square(n)))


class Laplace(Sampler):
    def __call__(self, dim):
        return np.random.laplace(0, SQRT05, size=dim)


class Logistic(Sampler):
    def __call__(self, dim):
        return np.random.logistic(0, SQRT3pi, size=dim)
    
class dWeibull(Sampler):
    def __call__(self, dim):
        return stats.dweibull(2.0, scale=1).rvs(size=dim)
    
    
class Cauchy(Sampler):
    def __call__(self, dim):
        sample = stats.cauchy(scale=1).rvs(size=dim)
        while not np.all(np.isfinite(sample)):
            sample = stats.cauchy(scale=1).rvs(size=dim)
        return sample.clip(-1e10, 1e10)
    
class ShiftedNormal(Sampler):
    def __call__(self, dim):
        return np.random.normal(0.5, 1, size=dim)
