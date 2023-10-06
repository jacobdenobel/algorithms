from dataclasses import dataclass

import numpy as np
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET


@dataclass
class MAES(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    lambda_: int = None
    mu: float = None
    sigma0: float = 1
    verbose: bool = True

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        n = problem.meta_data.n_variables
        self.lambda_ = self.lambda_ or (4 + np.floor(3 * np.log(n))).astype(int)
        self.mu = self.lambda_ // 2

        echi = np.sqrt(n) * (1 - (1 / n / 4) - (1 / n / n / 21))
        wi_raw = np.log(self.lambda_ / 2 + 0.5) - np.log(np.arange(1, self.mu + 1))
        w = wi_raw / np.sum(wi_raw)
        mueff = 1 / np.sum(np.power(w, 2))
        c_s = (mueff + 2) / (n + mueff + 5)
        c_1 = 2 / (pow(n + 1.3, 2) + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + (1 / mueff)) / (pow(n + 2, 2) + mueff))
        d_s = 1 + c_s + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1)
        sqrt_s = np.sqrt(c_s * (2 - c_s) * mueff)

        x = np.random.uniform(problem.bounds.lb, problem.bounds.ub).reshape(-1, 1)
        sigma = self.sigma0
        M = np.eye(n)
        s = np.ones((n, 1))

        while self.not_terminate(problem, self.lambda_):
            Z = np.random.normal(0, 1, (n, self.lambda_))
            D = M.dot(Z)
            X = x + (sigma * D)
            f = np.array(problem(X.T))
            idx = np.argsort(f)
            mu_best = idx[: self.mu]

            z = np.sum(w * Z[:, mu_best], axis=1, keepdims=True)
            d = np.sum(w * D[:, mu_best], axis=1, keepdims=True)
            x = x + (sigma * d)
            s = ((1 - c_s) * s) + (sqrt_s * z)

            M = (
                ((1 - 0.5 * c_1 - 0.5 * c_mu) * M)
                + ((0.5 * c_1) * M.dot(s).dot(s.T))
                + ((0.5 * c_mu * w) * D[:, mu_best]).dot(Z[:, mu_best].T)
            )
            sigma = sigma * np.exp(c_s / d_s * (np.linalg.norm(s) / echi - 1))

            if self.verbose:
                print(problem.state)
