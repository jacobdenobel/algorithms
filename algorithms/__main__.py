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
from .utils import ert, rastrigin, get_meshgrid, plot_contour
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


if __name__ == "__main__":
    parsert = ArgumentParser()
    parsert.add_argument("-f", "--fid", type=int, default=8)
    parsert.add_argument("-d", "--dim", type=int, default=2)
    parsert.add_argument("-i", "--iterations", type=int, default=1)
    parsert.add_argument("--logged", action="store_true")
    parsert.add_argument("--full-bbob", action="store_true")
    parsert.add_argument("--rastrigin", action="store_true")
    parsert.add_argument("--plot", action="store_true")
    args = parsert.parse_args()

    budget = args.dim * 5e4
    
    result_string = (
        "FCE:\t{:10.8f}\t{:10.4f}\n"
        "ERT:\t{:10.4f}\t{:10.4f}\n"
        "{}/{} runs reached target"
    )
    fids = [args.fid]
    if args.full_bbob:
        fids = list(range(1, 25))

    for alg in (
        # OrthogonalES(10_000),
        # CSA(budget),
        # CoordinateDescent(),
        # NelderMead(),
        # SPSA(budget),
        # GuidedESV2(budget),
        # GuidedES(budget),
        # GuidedES(budget),
        CMAES(budget, verbose=False, old_lr=True),
        CholeskyCMAES(budget),
        OnePlusOneCholeskyCMAES(budget),
        OnePlusOneCMAES(budget),
        # DR1(budget, verbose=False),
        # DR2(budget, verbose=False),
        # EGS(budget),
        # ARSV1(budget),
        # DR3(budget, verbose=False),
    ):
        
        
        alg_name = alg.__class__.__name__
        if args.logged:
            logger = ioh.logger.Analyzer(
                algorithm_name=alg_name, root="data", folder_name=alg_name
            )
        for fid in fids:
            np.random.seed(10)
            
            if args.rastrigin:
                problem = ioh.wrap_problem(
                    rastrigin,
                    "rastrigin",
                    dimension=args.dim,
                    ub=5,
                    lb=-5,
                    calculate_objective=lambda dim, iid: ([0] * dim, 0),
                )
            else:
                problem = ioh.get_problem(fid, 1, args.dim)


            if args.plot and args.dim == 2:
                X, Y, Z = get_meshgrid(problem, -5, 5)
                plot_contour(X, Y, Z)
                plt.show()
                problem.reset()
                
            if args.logged:
                problem.attach_logger(logger)
            
            fopts = []
            evals = []
            n_succ = 0
            print(
                f"Running {args.iterations} reps with {alg_name} on {problem}"
            )
            for i in range(args.iterations):
                alg(problem)
                fopts.append(problem.state.current_best.y)
                evals.append(problem.state.evaluations)
                n_succ += (
                    problem.state.current_best.y - problem.optimum.y
                ) < alg.target
                problem.reset()

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
