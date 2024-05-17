import warnings
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


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
            self.wi_raw = np.log(self.lambda_ / 2 + 0.5) - np.log(
                np.arange(1, self.mu + 1)
            )
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


def plot_contour(X, Y, Z, colorbar=True):
    plt.contourf(
        X, Y, np.log10(Z), levels=200, cmap="Spectral", zorder=-1, vmin=-2.5, vmax=2.5
    )
    plt.xlabel(R"$x_1$")
    plt.ylabel(R"$x_2$")
    if colorbar:
        plt.colorbar()
    plt.tight_layout()


def get_meshgrid(objective_function, lb, ub, delta: float = 0.025, z_tol=1e-8):
    x = np.arange(lb, ub + delta, delta)
    y = np.arange(lb, ub + delta, delta)

    if hasattr(objective_function, "optimum"):
        xo, yo = objective_function.optimum.x
        x = np.sort(np.r_[x, xo])
        y = np.sort(np.r_[y, yo])

    X, Y = np.meshgrid(x, y)

    Z = np.zeros(X.shape)
    for idx1 in range(X.shape[0]):
        for idx2 in range(X.shape[1]):
            Z[idx1, idx2] = objective_function([X[idx1, idx2], Y[idx1, idx2]])

    if hasattr(objective_function, "optimum"):
        Z = Z - objective_function.optimum.y

    if z_tol is not None:
        Z = Z.clip(z_tol)

    return X, Y, Z


def sphere(x: np.ndarray) -> float:
    assert x.ndim == 1
    return float(x.dot(x))


def rastrigin(x: np.ndarray) -> float:
    cosi = float(np.sum(np.cos(2 * np.pi * x)))
    return float(10 * (len(x) - cosi) + sphere(x))


def rosenbrock(x: np.ndarray) -> float:
    x_m_1 = x[:-1] - 1
    x_diff = x[:-1] ** 2 - x[1:]
    return float(100 * x_diff.dot(x_diff) + x_m_1.dot(x_m_1))


def lunacek(x: np.ndarray) -> float:
    problemDimensions = len(x)
    s = 1.0 - (1.0 / (2.0 * np.sqrt(problemDimensions + 20.0) - 8.2))
    mu1 = 2.5
    mu2 = -np.sqrt(abs((mu1**2 - 1.0) / s))
    firstSum = 0.0
    secondSum = 0.0
    thirdSum = 0.0
    for i in range(problemDimensions):
        firstSum += (x[i] - mu1) ** 2
        secondSum += (x[i] - mu2) ** 2
        thirdSum += 1.0 - np.cos(2 * np.pi * (x[i] - mu1))
    return min(firstSum, 1.0 * problemDimensions + secondSum) + 10 * thirdSum
