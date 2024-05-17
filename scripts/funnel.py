import os
from argparse import ArgumentParser
from itertools import product
from multiprocessing import Pool, cpu_count
from functools import wraps

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import ioh
import numpy as np
import matplotlib.pyplot as plt

from algorithms import utils, CoordinateDescent, CMAES

DATA = os.path.realpath(os.path.join(os.path.dirname(__file__), "../data"))


def run_parallel_function(runFunction, arguments):
    arguments = list(arguments)
    p = Pool(min(12, len(arguments)))
    results = p.map(runFunction, arguments)
    p.close()
    return results

class ParallelFunction:
    def __init__(self, method, n_trials):
        self.n_trials = n_trials
        if method == "local_search":
            self.alg = CoordinateDescent(100_000)
        elif method == "cma_es":
            self.alg = CMAES(100_000)
        elif method.startswith("cma_es"):
            self.alg = CMAES(100_000, lambda_= int(method.split("cma_es")[-1]))
        else:
            raise TypeError()

    def __call__(self, x):
        d, s = x
        n = 30
        problem = ioh.problem.DoubleSphere(n, d, s)
        n_succ = 0
        for run in range(self.n_trials):
            self.alg(problem)
            n_succ += problem.state.final_target_found
            problem.reset()
            
        print(
            f"d:{d:.2f}, s:{s:.2f} success ratio: {n_succ}/{self.n_trials} = {n_succ / self.n_trials:.3f}"
        )
        return (d, s, n_succ, self.n_trials, n_succ / self.n_trials)


def run_double_sphere(method="local_search", n_trials=1000):
    filename = os.path.join(DATA, f"{method}_sr.npy")
    if os.path.isfile(filename):
        data = np.load(filename)
    else:
        np.random.seed(2)
        parameters = product((1.0, 2.0, 3.0), np.arange(0.2, 1.5, 0.1))
        data = run_parallel_function(ParallelFunction(method, n_trials), parameters)
        data = np.array(data)
        np.save(filename, data)

    return data


def plot_s_vs_success_rate(data, name="LS"):
    plt.figure()
    for d in (1.0, 2.0, 3.0):
        pdata = data[data[:, 0] == d]
        plt.plot(pdata[:, 1], pdata[:, 4], label=f"d = {d}")
    plt.grid()
    plt.xlabel("s")
    plt.ylabel(r"$\hat\omega_{%s}$" % name)
    plt.legend()
    
def plot_vs_local_search(data, data_ls, title):
    plt.figure()
    sr_ls = sorted(data_ls[:, -1])
    plt.plot(sr_ls, sr_ls, label="LS")
    for d in (1.0, 2.0, 3.0):
        pdata = data[data[:, 0] == d][:, [1, 4]]
        pdata_ls = data_ls[data_ls[:, 0] == d][:, [1, 4]]
        plt.plot(pdata_ls[:, 1], pdata[:, 1], label=f"d = {d}")
    plt.title(title)
    plt.grid()
    plt.xlim(data_ls[:, -1].min(), data_ls[:, -1].max())
    plt.ylabel(r"$\hat\omega$")
    plt.xlabel(r"$\hat\omega_{LS}$")
    plt.legend()
    


if __name__ == "__main__":
    parsert = ArgumentParser()
    parsert.add_argument("-n", "--n_variables", type=int, default=2)
    parsert.add_argument("-d", "--depth", type=float, default=0.0)
    parsert.add_argument("-s", "--size", type=float, default=1.0)
    parsert.add_argument("-i", "--iterations", type=int, default=25)
    parsert.add_argument("--rastrigin", action="store_true")
    parsert.add_argument("--plot", action="store_true")
    args = parsert.parse_args()
    
    cma_es500 = run_double_sphere(method="cma_es500")
    cma_es = run_double_sphere(method="cma_es")
    local_search = run_double_sphere(method="local_search")
    
    plot_s_vs_success_rate(local_search)
    plot_vs_local_search(cma_es, local_search, title=r"CMA-ES $\lambda = 14$")
    plot_vs_local_search(cma_es500, local_search, title=r"CMA-ES $\lambda = 500$")
    plt.show()

    exit(0)
    if args.rastrigin:
        problem = ioh.problem.DoubleRastrigin(args.n_variables, args.depth, args.size)
    else:
        problem = ioh.problem.DoubleSphere(args.n_variables, args.depth, args.size)
    print(problem)

    if args.plot:
        X, Y, Z = utils.get_meshgrid(problem, -5, 5)
        utils.plot_contour(X, Y, Z)
        plt.show()
