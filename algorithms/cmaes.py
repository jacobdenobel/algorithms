from dataclasses import dataclass

import numpy as np
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET


@dataclass
class CMAES(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    lambda_: int = None
    mu: float = None
    sigma0: float = 2.0
    verbose: bool = True
    sep: bool = False

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        n = problem.meta_data.n_variables
        self.lambda_ = self.lambda_ or (4 + np.floor(3 * np.log(n))).astype(int)
        self.mu = self.lambda_ // 2
        sigma = self.sigma0
        # w
        w = np.log((self.lambda_ + 1) / 2) - np.log(np.arange(1, self.lambda_ + 1))
        w = w[: self.mu]
        mueff = w.sum() ** 2 / (w**2).sum()
        w = w / w.sum()

        # Learning rates
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (1 / 4 + mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + 2 * mueff / 2))
        cc = (4 + (mueff / n)) / (n + 4 + (2 * mueff / n))
        cs = (mueff + 2) / (n + mueff + 5)
        damps = 1.0 + (2.0 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs)
        chiN = n**0.5 * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        # dynamic parameters
        m = np.random.rand(n, 1)
        dm = np.zeros(n)
        pc = np.zeros((n, 1))
        ps = np.zeros((n, 1))
        B = np.eye(n)
        C = np.eye(n)
        D = np.ones((n, 1))
        invC = np.eye(n)

        while not self.should_terminate(problem, self.lambda_):
            Z = np.random.normal(0, 1, (n, self.lambda_))
            Y = np.dot(B, D * Z)
            X = m + (sigma * Y)
            f = np.array(problem(X.T))

            # select
            fidx = np.argsort(f)
            mu_best = fidx[: self.mu]

            # recombine
            m_old = m.copy()
            m = m_old + (1 * ((X[:, mu_best] - m_old) @ w).reshape(-1, 1))

            # adapt
            dm = (m - m_old) / sigma
            ps = (1 - cs) * ps + (np.sqrt(cs * (2 - cs) * mueff) * invC @ dm)
            sigma *= np.exp((cs / damps) * ((np.linalg.norm(ps) / chiN) - 1))
            hs = (
                np.linalg.norm(ps)
                / np.sqrt(1 - np.power(1 - cs, 2 * (problem.state.evaluations / self.lambda_)))
            ) < (1.4 + (2 / (n + 1))) * chiN

            dhs = (1 - hs) * cc * (2 - cc)
            pc = (1 - cc) * pc + (hs * np.sqrt(cc * (2 - cc) * mueff)) * dm


            rank_one = c1 * pc * pc.T
            old_C = (1 - (c1 * dhs) - c1 - (cmu * w.sum())) * C
            rank_mu = cmu * (w * Y[:, mu_best] @ Y[:, mu_best].T)
            C = old_C + rank_one + rank_mu

            if np.isinf(C).any() or np.isnan(C).any() or (not 1e-16 < sigma < 1e6):
                sigma = self.sigma0
                pc = np.zeros((n, 1))
                ps = np.zeros((n, 1))
                C = np.eye(n)
                B = np.eye(n)
                D = np.ones((n, 1))
                invC = np.eye(n)
            else:
                C = np.triu(C) + np.triu(C, 1).T
                if not self.sep:
                    D, B = np.linalg.eigh(C)
                else:
                    D = np.diag(C)


            D = np.sqrt(D).reshape(-1, 1)
            invC = np.dot(B, D ** -1 * B.T)
