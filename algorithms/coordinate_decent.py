import math
from dataclasses import dataclass

import numpy as np
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET

TOL = 1e-8
INV_PHI = (np.sqrt(5) - 1) / 2  # 1 / phi
INV_PHI2 = (3 - np.sqrt(5)) / 2  # 1 / phi^2


def golden_section_search(fi, a=-1, b=1, tol=TOL, middle=True):
    """
    Perform golden section search to find the minimum of a 1D function fi.

    Given a function fi with a single local minimum in the interval [a,b],
    the function returns a subset interval [c,d] that contains
    the minimum with d-c <= tol. If middle is True, the method returns
    the middle point of [c, d].

    Parameters
    ----------
    fi: function
        The 1D function to minimize.
    a: float
        Lower bound of the search interval.
    b: float
        Upper bound of the search interval.
    tol :float
        Tolerance for the search.

    Returns:
        [c, d]: (int, int)
            The interval that contains the minimum
        x_min: float
            The argument that minimizes the function fi within [a, b]
    """

    (a, b) = (min(a, b), max(a, b))
    h = b - a
    if h <= tol:
        return (a, b)

    # Required steps to achieve tolerance
    n = int(math.ceil(math.log(tol / h) / math.log(INV_PHI)))

    c = a + INV_PHI2 * h
    d = a + INV_PHI * h
    yc = fi(c)
    yd = fi(d)

    for _ in range(n - 1):
        if yc < yd:  # yc > yd to find the maximum
            b = d
            d = c
            yd = yc
            h = INV_PHI * h
            c = a + INV_PHI2 * h
            yc = fi(c)
        else:
            a = c
            c = d
            yc = yd
            h = INV_PHI * h
            d = a + INV_PHI * h
            yd = fi(d)

    if yc < yd:
        c, d = a, d
    else:
        c, d = c, b

    if middle:
        return (c + d) / 2
    return c, d


def make_phi(f, x, i):
    """Make a function which optimizes f along the i-th coordinate of x"""

    def phi(xi):
        x[i] = xi
        return f(x)

    return phi


@dataclass
class CoordinateDescent(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    tol: float = TOL
    x0: np.ndarray = None

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        x_current = np.random.uniform(problem.bounds.lb, problem.bounds.ub) \
            if self.x0 is None else self.x0.copy()

        while self.budget > problem.state.evaluations:
            x_prev = x_current.copy()
            x_copy = x_prev.copy()
            
            for i in range(problem.meta_data.n_variables):
                x_current[i] = golden_section_search(
                    make_phi(problem,  x_copy, i),
                    problem.bounds.lb[i] - TOL,
                    problem.bounds.ub[i] + TOL,
                )

            # Check for convergence
            if np.linalg.norm(x_current - x_prev) < self.tol:
                break

        return problem.state.current.y, problem.state.current.x
