import warnings
from dataclasses import dataclass
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
    

def init_lambda(n, method="default", even=True):
    """
        range:      2*mu < lambda < 2*n + 10 
        default:    4 + floor(3 * ln(n))     

    """
    if method == "default":
        lamb = (4 + np.floor(3 * np.log(n))).astype(int) 
    elif method == "n/2":
        lamb = max(32, np.floor(n / 2).astype(int))
    else:
        raise ValueError()
    if even and lamb % 2 != 0:
        lamb += 1
    return lamb
    
