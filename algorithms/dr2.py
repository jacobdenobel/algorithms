from dataclasses import dataclass

import numpy as np
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET


@dataclass
class DR2(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    mu: int = 1
    lambda_: int = 10
    sigma0: float = 1
    verbose: bool = True
    mirrored: bool = True
    bound_correct: bool = False
    init: str = "same"

    def get_weights(self, mu, p=.25):
        w = 1 / np.power(np.arange(1, mu+1), 2)
        # w = mu / (2 ** np.arange(1, mu + 1)) + ((1 / (2 ** mu)) / mu)
        # w = np.power(w, p)
        w = w / w.sum()
        e = (1 - w.sum()) / len(w)
        return (w + e).reshape(-1, 1)


    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        dim = problem.meta_data.n_variables
        beta_scale = 1 / dim
        
        beta = np.sqrt(beta_scale)
        c = beta

        n_samples = self.lambda_
        if self.mirrored:
            if self.lambda_ % 2 != 0:
                self.lambda_ += 1
            n_samples = int(self.lambda_ // 2)

        zeta = np.zeros((dim, 1))
        sigma_local = np.ones((dim, 1)) * self.sigma0
        sigma = self.sigma0

        c1 = np.sqrt(c / (2 - c))
        c2 = np.sqrt(dim) * c1


        if isinstance(self.init, np.ndarray):
            x_prime = self.init.copy()
        elif self.init == "zeros":
            x_prime = np.zeros((dim, 1))
        elif self.init == "same":
            x_prime = np.ones((dim, 1)) / dim
        else:
            x_prime = np.random.uniform(problem.bounds.lb, problem.bounds.ub, (dim, 1))

        w = self.get_weights(self.mu)
        try:
            while not self.should_terminate(problem, self.lambda_):
                z = np.random.normal(size=(dim, n_samples))
                if self.mirrored:
                    z = np.c_[z, -z]

                y = sigma * sigma_local * z
                x = x_prime + y

                if self.bound_correct:
                    x = x.clip(problem.bounds.lb, problem.bounds.ub)

                # f = np.array([problem(xi) for xi in x])
                f = problem(x.T)

                idx = np.argsort(f)[:self.mu]

                x_prime = x_prime + (y[:, idx] @ w)
                z_prime = z[:, idx] @ w

                zeta = ((1 - c) * zeta) + (c * z_prime)
                sigma *= np.power(
                    np.exp((np.linalg.norm(zeta) / c2) - 1 + (1 / (5 * dim))), beta
                )
                sigma_local *= np.power((np.abs(zeta) / c1) + (7 / 20), beta_scale)

                if self.verbose:
                    print(
                        f"e: {problem.state.evaluations}/{self.budget}",
                        f"fopt: {problem.state.current_best.y:.3f};",
                        f"f: {np.median(f):.3f} +- {np.std(f):.3f} ",
                        f"[{np.min(f):.3f}, {np.max(f):.3f}];",
                        f"sigma: {sigma:.3e}",
                        f"sigma_local: {np.median(sigma_local):.3e} +- {np.std(sigma_local):.3f};",
                    )
        except KeyboardInterrupt:
            pass
        return x_prime
