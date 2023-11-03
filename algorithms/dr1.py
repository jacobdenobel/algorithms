from dataclasses import dataclass

import numpy as np
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET


@dataclass
class DR1(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    lambda_: int = 10
    sigma0: float = 1e-1
    greedy: bool = False
    verbose: bool = True

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        dim = problem.meta_data.n_variables
        beta_scale = 1 / dim
        beta = np.sqrt(beta_scale)
        zeta = np.array([5 / 7, 7 / 5])
        sigma = np.ones((dim, 1)) * self.sigma0

        root_pi = np.sqrt(2 / np.pi)

        x_prime = np.random.uniform(problem.bounds.lb, problem.bounds.ub)
        fbest = float("inf")
        while not self.should_terminate(problem, self.lambda_):
            z = np.random.normal(size=(dim, self.lambda_))
            zeta_i = np.random.choice(zeta, self.lambda_)

            x = x_prime + (zeta_i * (sigma * z)).T
            f = problem(x)

            idx = np.argmin(f)
            if self.greedy and f[idx] < fbest:
                fbest = f[idx]
                x_prime = x[idx, :].copy()
            elif not self.greedy:
                x_prime = x[idx, :].copy()

            zeta_sel = np.exp(np.abs(z[:, idx]) - root_pi)
            sigma *= (
                np.power(zeta_i[idx], beta) * np.power(zeta_sel, beta_scale)
            ).reshape(-1, 1)

            if self.verbose:
                print(
                    f"e: {problem.state.evaluations}/{self.budget}",
                    f"fopt: {problem.state.current_best.y:.3f};",
                    f"f: {np.median(f):.3f} +- {np.std(f):.3f} ",
                    f"[{np.min(f):.3f}, {np.max(f):.3f}];",
                    f"sigma_local: {np.median(sigma):.3e} +- {np.std(sigma):.3f};",
                )
        return x_prime
