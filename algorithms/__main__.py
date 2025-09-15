import itertools
from time import sleep
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count

import ioh
import numpy as np
import matplotlib.pyplot as plt

from .maes import MAES
from .cmaes import CMAES
from .dr1 import DR1
from .dr2 import DR2
from .dr3 import DR3
from .egs import EGS
from .ars import ARSV1
from .utils import ert, get_meshgrid
from .ges import GuidedES, GuidedESV2
from .spsa import SPSA
from .csa_grad import CSAGrad
from .coordinate_decent import CoordinateDescent
from .ortho_es import OrthogonalES
from .csa import CSA
from .nelder_mead import NelderMead
from .cholesky_cma import CholeskyCMAES
from .one_plus_one_cma import OnePlusOneCholeskyCMAES, OnePlusOneCMAES
from .lmmaes import LMMAES
from .one_plus_one_es import OnePlusOneES
from .evolution_strategy import EvolutionStrategy
from .mu_comma_lambda_sa_es import SaEvolutionStrategy
from .sampling import *

result_string = (
    "FCE:\t{:10.8f}\t{:10.4f}\n"
    "ERT:\t{:10.4f}\t{:10.4f}\n"
    "{}/{} runs reached target"
)

def run_alg(alg, args, fids, iids, dims):
    alg_name = str(alg)
    if args.logged:
        logger = ioh.logger.Analyzer(
            algorithm_name=alg_name,
            root="data",
            folder_name=alg_name,
            store_positions=args.store_positions,
        )

    for fid, iid, dim in itertools.product(fids, iids, dims):
        alg.budget = dim * 1e5

        if args.logged and hasattr(alg, "sigma"):
            logger.watch(alg, "sigma")

        problem = ioh.get_problem(fid, iid, dim)

        if args.logged:
            problem.attach_logger(logger)

        fopts = []
        evals = []
        n_succ = 0
        for i in range(args.iterations):
            np.random.seed(args.seed + i * 7)
            alg(problem)
            fopts.append(problem.state.current_best.y)
            evals.append(problem.state.evaluations)
            n_succ += (problem.state.current_best.y - problem.optimum.y) < alg.target
            problem.reset()

        print(f"Completed {args.iterations} reps with {alg_name} on {problem}")
        print(
            result_string.format(
                np.mean(fopts),
                np.std(fopts),
                *ert(evals, n_succ),
                n_succ,
                args.iterations,
            )
        )
        print()


if __name__ == "__main__":
    parsert = ArgumentParser()
    parsert.add_argument("-f", "--fid", type=int, default=None)
    parsert.add_argument("-d", "--dim", type=int, default=None)
    parsert.add_argument("-r", "--iterations", type=int, default=1)
    parsert.add_argument("-i", "--instances", type=int, default=1)
    parsert.add_argument("-s", "--seed", type=int, default=1)
    parsert.add_argument("--logged", action="store_true")
    parsert.add_argument("--store_positions", action="store_true")
    parsert.add_argument("--sequential", action="store_true")
    args = parsert.parse_args()

    if args.dim is None:
        dims = (2, 3, 5, 10, 20,)
    else:
        dims = [args.dim]

    iids = list(range(1, args.instances + 1))

    if args.fid is None:
        fids = list(range(1, 25))
    else:
        fids = [args.fid]

    algs = (
        # SaEvolutionStrategy(),
        # SaEvolutionStrategy(sampler=Uniform()),
        # SaEvolutionStrategy(sampler=Logistic()),
        # SaEvolutionStrategy(sampler=Laplace()),
        # SaEvolutionStrategy(sampler=dWeibull()),
        # OnePlusOneCMAES(),
        CMAES(),
    )
    
    
    def run_func(alg):
        return run_alg(alg, args, fids, iids, dims)
    
    if args.sequential:
        for alg in algs:
            run_func(alg)
    else:
        n_proc = min(cpu_count(), len(algs))
        with Pool(n_proc) as p:
            p.map(run_func, algs)
    
