import warnings
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