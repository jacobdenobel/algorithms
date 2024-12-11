import itertools
from time import sleep
from argparse import ArgumentParser
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
from .utils import ert, rastrigin, get_meshgrid
# from .sges import SalimansES, GuidedES, SelfGuidedES
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
from .sampling import *

if __name__ == "__main__":
    parsert = ArgumentParser()
    parsert.add_argument("-f", "--fid", type=int, default=8)
    parsert.add_argument("-d", "--dim", type=int, default=2)
    parsert.add_argument("-r", "--iterations", type=int, default=1)
    parsert.add_argument("-i", "--instances", type=int, default=1)
    parsert.add_argument("--logged", action="store_true")
    parsert.add_argument("--full-bbob", action="store_true")
    parsert.add_argument("--plot", action="store_true")
    args = parsert.parse_args()

    # dims = (args.dim, )
    dims = (2, 5, 10, 20)
    
    result_string = (
        "FCE:\t{:10.8f}\t{:10.4f}\n"
        "ERT:\t{:10.4f}\t{:10.4f}\n"
        "{}/{} runs reached target"
    )
    fids = [args.fid]
    
    iids = list(range(1, args.instances + 1))
    
    if args.full_bbob:
        fids = list(range(1, 25))
    
    sleep(np.random.uniform(0, 2))
    
    for alg in (
        # OrthogonalES(10_000),
        # CSA(budget),
        # CoordinateDescent(),
        # NelderMead(),
        # SPSA(budget),
        # GuidedESV2(budget),
        # GuidedES(budget),
        # GuidedES(budget),
        # CMAES(budget, verbose=False, old_lr=True),
        # LMMAES(),
        # CholeskyCMAES(),
        CholeskyCMAES(),
        CholeskyCMAES(sampler=Uniform()),
        # CholeskyCMAES(sampler=Logistic()),
        # CholeskyCMAES(sampler=Laplace()),
        # OnePlusOneCMAES(),
        # DR1(budget, verbose=False),
        # DR2(budget, verbose=False),
        # EGS(budget),
        # ARSV1(budget),
        # DR3(budget, verbose=False),
    ):
        alg_name = str(alg)
        if args.logged:
            logger = ioh.logger.Analyzer(
                algorithm_name=alg_name, 
                root="data", 
                folder_name=alg_name,
                store_positions=True
            )
        for fid, iid, dim in itertools.product(fids, iids, dims):
            alg.budget = dim * 1e5
            np.random.seed(11)
            
            if args.logged and hasattr(alg, 'sigma'):
                logger.watch(alg, 'sigma')
                
            problem = ioh.get_problem(fid, iid, dim)

            if args.logged:
                problem.attach_logger(logger)
            
            fopts = []
            evals = []
            n_succ = 0
            for i in range(args.iterations):
                alg(problem)
                fopts.append(problem.state.current_best.y)
                evals.append(problem.state.evaluations)
                n_succ += (
                    problem.state.current_best.y - problem.optimum.y
                ) < alg.target
                problem.reset()

            print(
                f"Completed {args.iterations} reps with {alg_name} on {problem}"
            )
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
