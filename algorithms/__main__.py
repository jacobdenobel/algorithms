from argparse import ArgumentParser
import ioh
import numpy as np
from .maes import MAES
from .cmaes import CMAES
from .dr1 import DR1
from .dr2 import DR2
from .dr3 import DR3
from .utils import ert


if __name__ == "__main__":
    parsert = ArgumentParser()
    parsert.add_argument("-f", "--fid", type=int, default=1)
    parsert.add_argument("-d", "--dim", type=int, default=5)
    parsert.add_argument("-i", "--iterations", type=int, default=25)
    parsert.add_argument("--logged", action="store_true")
    parsert.add_argument("--full-bbob", action="store_true")
    args = parsert.parse_args()

    budget = args.dim * 1e4

    result_string = (
        "FCE:\t{:10.8f}\t{:10.4f}\n"
        "ERT:\t{:10.4f}\t{:10.4f}\n"
        "{}/{} runs reached target"
    )
    fids = [args.fid]
    if args.full_bbob:
        fids = list(range(1, 25))

    for alg in (
        CMAES(budget, verbose=False), 
        MAES(budget, verbose=False), 
        # DR1(budget, verbose=False),
        # DR2(budget, verbose=False),
        # DR3(budget, verbose=False),
    ):
        alg_name = alg.__class__.__name__
        if args.logged:
            logger = ioh.logger.Analyzer(
                algorithm_name=alg_name, 
                root="data",
                folder_name=alg_name
            )
        for fid in fids:
            np.random.seed(10)
            problem = ioh.get_problem(fid, 1, args.dim)
            if args.logged:
                problem.attach_logger(logger)

            fopts = []
            evals = []
            n_succ = 0
            print(f"Running {args.iterations} reps with {alg_name} on f{fid} in {args.dim}D")
            for i in range(args.iterations):
                alg(problem)
                fopts.append(problem.state.current_best.y)
                evals.append(problem.state.evaluations)
                n_succ += (problem.state.current_best.y - problem.optimum.y) < alg.target
                problem.reset()

            print(
                result_string.format(
                    np.mean(fopts),
                    np.std(fopts),
                    *ert(evals, n_succ),
                    n_succ, args.iterations,
                )
            )
            print()
