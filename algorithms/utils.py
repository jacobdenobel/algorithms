import warnings
from dataclasses import dataclass
from scipy import qmc

import numpy as np
    
def ert(evals, n_succ):
    """Computed the expected running time of a list of evaluations.

    Parameters
    ----------
    evals: list
        a list of running times (number of evaluations)
    budget: int
        the maximum number of evaluations

    Returns
    -------
    float
        The expected running time

    float
        The standard deviation of the expected running time
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            evals = np.array(evals)
            _ert = float(evals.sum()) / int(n_succ)
        return _ert, np.std(evals)
    except ZeroDivisionError:
        return float("inf"), np.nan



@dataclass
class Weights:
    mu: int
    lambda_: int
    n: int
    method: str = "log"

    def __post_init__(self):
        self.set_weights()
        self.normalize_weights()

    def set_weights(self):
        if self.method == "log":
            self.wi_raw = np.log(self.lambda_ / 2 + 0.5) - np.log(np.arange(1, self.mu + 1))
        elif self.method == "linear":
            self.wi_raw = np.arange(1, self.mu + 1)[::-1]
        elif self.method == "equal":
            self.wi_raw = np.ones(self.mu)

    def normalize_weights(self):
        self.w = self.wi_raw / np.sum(self.wi_raw) 
        self.w_all = np.r_[self.w, -self.w[::-1]]

    @property
    def mueff(self):
        return 1 / np.sum(np.power(self.w, 2))

    @property
    def c_s(self):
        return (self.mueff + 2) / (self.n + self.mueff + 5)
    
    @property
    def d_s(self):
        return 1 + self.c_s + 2 * max(0, np.sqrt((self.mueff - 1) / (self.n + 1)) - 1)
    
    @property
    def sqrt_s(self):
        return np.sqrt(self.c_s * (2 - self.c_s) * self.mueff)
    

def init_lambda(n, method="n/2"):
    """
        range:      2*mu < lambda < 2*n + 10 
        default:    4 + floor(3 * ln(n))     

    """
    if method == "default":
        return (4 + np.floor(3 * np.log(n))).astype(int) 
    
    elif method == "n/2":
        return max(32, np.floor(n / 2).astype(int))
    else:
        raise ValueError()
    


@dataclass
class Initializer:
    n: int 
    lb: float = -0.1
    ub: float =  0.1
    method: str = "lhs"
    fallback: str = "zero"
    n_evals: int = 0
    max_evals: int = 500
    max_observed: float = -np.inf
    min_observed: float =  np.inf

    def __post_init__(self):
        self.sampler = qmc.LatinHypercube(self.n)

    def static_init(self, method):
        if method == "zero":
            return np.zeros((self.n, 1))
        elif method == "uniform":
            return np.random.uniform(self.lb, self.ub, size=(self.n, 1))
        elif method == "gauss":
            return np.random.normal(size=(self.n, 1))
        raise ValueError()

    def get_x_prime(self, problem, samples_per_trial: int = 10) -> np.ndarray:
        if self.method != "lhs":
            return self.static_init(self.method)

        samples = None
        sample_values = np.array([])
        f = np.array([0])
        while self.n_evals < self.max_evals:
            X = qmc.scale(self.sampler.random(samples_per_trial), self.lb, self.ub).T
            f = problem(X)
            self.n_evals += samples_per_trial
            self.max_observed = max(self.max_observed, f.max())
            self.min_observed = max(self.min_observed, f.max())
            
            if f.std() > 0:
                idx = f != self.max_observed
                if samples is None:
                    samples = X[:, idx]
                else:
                    samples = np.c_[samples, X[:, idx]]
                sample_values = np.r_[sample_values, f[idx]]
        
        if not any(sample_values):
            warnings.warn(f"DOE did not find any variation after max_evals={self.max_evals}"
                          f", using fallback {self.fallback} intialization.")
            return self.static_init(self.fallback)

        w = np.log(len(sample_values) + 0.5) - np.log(np.arange(1, len(sample_values) + 1))
        w = w / w.sum()
        idx = np.argsort(sample_values)
        x_prime = np.sum(w * samples[:, idx], axis=1, keepdims=True)
        return x_prime