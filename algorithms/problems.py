import ioh
import numpy as np


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
    n = len(x)
    s = 1.0 - (1.0 / (2.0 * np.sqrt(n + 20.0) - 8.2))
    mu1 = 2.5
    mu2 = -np.sqrt(abs((mu1**2 - 1.0) / s))
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    for i in range(n):
        s1 += (x[i] - mu1) ** 2
        s2 += (x[i] - mu2) ** 2
        s3 += 1.0 - np.cos(2 * np.pi * (x[i] - mu1))
    return min(s1, 1.0 * n + s2) + 10 * s3





