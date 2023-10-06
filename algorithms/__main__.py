from argparse import ArgumentParser
import ioh
import numpy as np
from .genetic_algorithm import GeneticAlgorithm
from .unbounded_integer_ea import DiscreteBBOB
from .maes import MAES
from .cmaes import CMAES
from .dr1 import DR1
from .dr2 import DR2

def ert(evals, budget):
    import warnings
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
    int
        The number of successful runs

    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            evals = np.array(evals)
            n_succ = (evals < budget).sum()
            _ert = float(evals.sum()) / int(n_succ)
        return _ert, np.std(evals), n_succ
    except ZeroDivisionError:
        return float("inf"), np.nan, 0

if __name__ == "__main__":
    parsert = ArgumentParser()
    parsert.add_argument("-f", "--fid", type=int, default=1)
    parsert.add_argument("-d", "--dim", type=int, default=5)
    args = parsert.parse_args()

    budget = args.dim * 1e4
    iterations = 10
    

    problem = ioh.get_problem(args.fid, 1, args.dim)
    result_string = (
        "FCE:\t{:10.8f}\t{:10.4f}\n"
        "ERT:\t{:10.4f}\t{:10.4f}\n"
        "{}/{} runs reached target"
    )

    for alg in (
        CMAES(budget, verbose=False), 
        MAES(budget, verbose=False), 
        DR1(budget, verbose=False),
        DR2(budget, verbose=False),
    ):
        np.random.seed(10)
        print(f"Running {iterations} reps with {alg.__class__.__name__}")
        fopts = []
        evals = []
        for i in range(iterations):
            alg(problem)
            fopts.append(problem.state.current_best.y)
            evals.append(problem.state.evaluations)
            problem.reset()

        print(
            result_string.format(
                np.mean(fopts),
                np.std(fopts),
                *ert(evals, budget),
                iterations,
            )
        )
        print()
