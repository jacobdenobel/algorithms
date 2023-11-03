from dataclasses import dataclass

import numpy as np
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET, SIGMA_MAX
from .utils import Weights, init_lambda


@dataclass
class DR1(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    mu: int = 1
    lambda_: int = 10
    sigma0: float = 1e-1
    verbose: bool = True
    mirrored: bool = True

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        n = problem.meta_data.n_variables
        self.lambda_ = self.lambda_ or init_lambda(n, "default")
        self.mu = self.mu or self.lambda_ // 2

        beta_scale = 1 / self.n
        beta = np.sqrt(beta_scale)
        zeta = np.array([5 / 7, 7 / 5])
        sigma = np.ones((self.n, 1)) * self.sigma0
        root_pi = np.sqrt(2 / np.pi)

        weights = Weights(self.mu, self.lambda_, self.n)
        x_prime = np.zeros((n, 1))
        n_samples = self.lambda_ if not self.mirrored else self.lambda_ // 2

        while not self.should_terminate(problem, self.lambda_):
            Z = np.random.normal(size=(self.n, n_samples))
            if self.mirrored:
                Z = np.hstack([Z, -Z])

            zeta_i = np.random.choice(zeta, (1, self.lambda_))
            Y = zeta_i * (sigma * Z)
            X = x_prime + Y
            f = problem(X)
            idx = np.argmin(f)

            mu_best = idx[: self.mu]

            y_prime = np.sum(Y[:, mu_best] * weights.w, axis=1, keepdims=True)
            x_prime = x_prime + y_prime

            z_prime = np.sum(
                Z[:, mu_best] * weights.w, axis=1, keepdims=True
            ) * np.sqrt(weights.mueff)
            zeta_w = np.sum(zeta_i[:, mu_best] * weights.w)
            zeta_sel = np.exp(np.abs(z_prime) - root_pi)
            sigma *= (np.power(zeta_w, beta) * np.power(zeta_sel, beta_scale)).reshape(
                -1, 1
            )
            sigma = sigma.clip(0, SIGMA_MAX)

            if self.verbose:
                print(
                    f"e: {problem.state.evaluations}/{self.budget}",
                    f"fopt: {problem.state.current_best.y:.3f};",
                    f"f: {np.median(f):.3f} +- {np.std(f):.3f} ",
                    f"[{np.min(f):.3f}, {np.max(f):.3f}];",
                    f"sigma_local: {np.median(sigma):.3e} +- {np.std(sigma):.3f};",
                )
        return x_prime
