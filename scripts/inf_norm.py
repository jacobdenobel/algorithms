import numpy as np
import ioh
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from algorithms.one_plus_one_es import OnePlusOneES
from algorithms.sampling import *
 
if __name__ == "__main__":
    
        
    for alg in (
        OnePlusOneES(),
        OnePlusOneES(sampler=Uniform()),
        OnePlusOneES(sampler=Logistic()),
        OnePlusOneES(sampler=Laplace()),
        OnePlusOneES(sampler=dWeibull()),
        OnePlusOneES(sampler=Cauchy()),
    ):
        n_runs = 10
        dims = (2, 10, 50)
        styles = {2: 'solid', 10: 'dashed', 50: 'dotted'}
        budget = max(dims) * 200

        sigmas = np.ones((len(dims), n_runs, budget))
        sigma_stars = np.ones((len(dims),n_runs, budget))
        dxs = np.zeros((len(dims), n_runs, budget))
        
        for di, dim in enumerate(dims):
            for run in range(n_runs):
                np.random.seed(run * 7)
                
                problem = ioh.get_problem(1, 1, dim)
                alg.restart(problem)
                for g in range(budget):
                    dx = np.linalg.norm(problem.state.current_best_internal.x)
                    sigmas[di, run, g] = alg.sigma
                    sigma_stars[di, run, g] = dx / np.sqrt(dim)
                    dxs[di, run, g] = dx
                    alg.step(problem)
            
            # ms = np.median(sigmas / sigma_stars, axis=0)
            # ss = np.std(sigmas / sigma_stars, axis=0)
            # plt.plot(ms)
        
        
            # plt.plot(ms, label=r'$\sigma / \sigma^*$')
        
        # s = np.median(sigmas, axis=1)
        # sp = np.median(sigma_stars, axis=1)
        print(alg.sampler)
        for i,d in enumerate(dims):
            x = sigmas #/ sigma_stars
            if i == 0:
                p = plt.loglog(10**np.mean(np.log10(x[i]), axis=0), linestyle=styles.get(d), label=alg.sampler)
                continue    
            plt.loglog(10**np.mean(np.log10(x[i]), axis=0), linestyle=styles.get(d), color=p[0].get_color())

    plt.grid()
    plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles + [
        Line2D([0], [0], color='black', label=r'$d=2$'),
        Line2D([0], [0], color='black', linestyle='dashed', label=r'$d=10$'),
        Line2D([0], [0], color='black', linestyle='dotted', label=r'$d=50$')
    ])
    plt.ylim(1e-10, 1e1)
    plt.xlim(1e1)
    plt.ylabel(r"$\sigma$")
    plt.xlabel("# evals")
    plt.show()
    # breakpoint()
    # plt.plot(np.median(np.median(sigmas / sigma_stars, axis=1), axis=0), label=alg.sampler)
    # plt.legend(title='Distribution')
    # plt.ylim(0, 2)
    # plt.grid()
    # plt.show()
    # breakpoint()
    # cmap = plt.cm.viridis.reversed()  # Example colormap
    # norm = mcolors.Normalize(vmin=dims.min(), vmax=dims.max())
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # for di, dim in enumerate(dims):
    #     p = plt.plot(s[di], label=f"{dim}D", c=cmap(norm(dim)))
    #     plt.plot(sp[di], linestyle='dashed', color=p[0].get_color())

    
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