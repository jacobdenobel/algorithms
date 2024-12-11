import numpy as np
import ioh
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from algorithms.one_plus_one_es import OnePlusOneES
from algorithms.sampling import *
 
if __name__ == "__main__":
    
        
    for alg in (OnePlusOneES(),
        OnePlusOneES(sampler=Uniform()),
        OnePlusOneES(sampler=Logistic()),
        OnePlusOneES(sampler=Laplace()),
    ):
        np.random.seed(1)
        
        
        n_runs = 10
        dims = np.arange(2, 50)
        budget = max(dims) * 50
        sigmas = np.ones((len(dims), n_runs, budget))
        sigma_stars = np.ones((len(dims),n_runs, budget))
        dxs = np.zeros((len(dims), n_runs, budget))
        
        for di, dim in enumerate(dims):
            for run in range(n_runs):
                problem = ioh.get_problem(1, 1, dim)
                alg.restart(problem)
                for g in range(budget):
                    dx = np.linalg.norm(problem.state.current_best_internal.x)
                    sigmas[di, run, g] = alg.sigma
                    sigma_stars[di, run, g] = dx / np.sqrt(dim)
                    dxs[di, run, g] = dx
                    alg.step(problem)
                    if dx < 1e-8:
                        break
            
            # ms = np.median(sigmas / sigma_stars, axis=0)
            # ss = np.std(sigmas / sigma_stars, axis=0)
            # plt.plot(ms)
        
        
            # plt.plot(ms, label=r'$\sigma / \sigma^*$')
        
        s = np.median(sigmas, axis=1)
        sp = np.median(sigma_stars, axis=1)
        
        
    
        plt.plot(np.median(np.median(sigmas / sigma_stars, axis=1), axis=0), label=alg.sampler)
    plt.legend(title='Distribution')
    plt.ylim(0, 2)
    plt.ylabel(r"$\sigma / \sigma^*$")
    plt.xlabel("# evals")
    plt.grid()
    plt.show()
    breakpoint()
    # cmap = plt.cm.viridis.reversed()  # Example colormap
    # norm = mcolors.Normalize(vmin=dims.min(), vmax=dims.max())
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # for di, dim in enumerate(dims):
    #     p = plt.plot(s[di], label=f"{dim}D", c=cmap(norm(dim)))
    #     plt.plot(sp[di], linestyle='dashed', color=p[0].get_color())

    # plt.legend(handles=[
    #     Line2D([0], [0], color='black', label=r'$\sigma$'),
    #     Line2D([0], [0], color='black', linestyle='dashed', label=r'$\sigma^*$')
        
    # ])
    
    # plt.colorbar(sm, ax=plt.gca(), label='dimensionality', ticks=dims)     
    # plt.ylabel(r"$\sigma / \sigma*$")
    # plt.xlabel("# evaluations")
    # plt.grid()
    # # plt.legend()
    # plt.yscale("log")
    # plt.ylim(1e-8, 1e2)
    # plt.show()
                 
        # plt.plot(np.median(sigmas, axis=0), label=r'$\sigma$')
        # plt.plot(np.median(sigma_stars, axis=0), label=r'$\sigma^*$')
        # plt.plot(np.median(dxs, axis=0), label=R'$||x - x^*||$')
        
        # plt.plot(sigmas, sigma_stars)
        # plt.show()
        # breakpoint()
        # break