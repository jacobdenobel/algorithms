from dataclasses import dataclass

import numpy as np
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET
from .utils import is_matrix_valid

@dataclass
class CMAES(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    lambda_: int = None
    mu: float = None
    sigma0: float = 2.0
    verbose: bool = True
    sep: bool = False
    old_lr: bool = False
    
    def restart(self, n):
        self.m = np.zeros((n, 1))
        self.pc = np.zeros((n, 1))
        self.ps = np.zeros((n, 1))
        self.B = np.eye(n)
        self.C = np.eye(n)
        self.D = np.ones((n, 1))
        self.invC = np.eye(n)
        self.sigma = self.sigma0

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        n = problem.meta_data.n_variables
        self.lambda_ = self.lambda_ or (4 + np.floor(3 * np.log(n))).astype(int)
        self.mu = self.lambda_ // 2
        # w
        w = np.log((self.lambda_ + 1) / 2) - np.log(np.arange(1, self.lambda_ + 1))
        w = w[: self.mu]
        w = w / w.sum()
        mueff = w.sum() ** 2 / (w**2).sum()

        # Learning rates
        if not self.old_lr:
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (1 / 4 + mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + 2 * mueff / 2))
            cc = (4 + (mueff / n)) / (n + 4 + (2 * mueff / n))
            cs = (mueff + 2) / (n + mueff + 5)
        else:
            cs = np.sqrt(mueff)/(np.sqrt(n) + np.sqrt(mueff))
            cc = 4.0 / (n + 4.0)
            ccov = 2.0 / np.square(n + np.sqrt(2.0))
        
        
        # Normalizers
        damps = 1.0 + (2.0 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs)
        chiN = n**0.5 * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        # dynamic parameters
        # m = np.random.uniform(problem.bounds.lb, problem.bounds.ub).reshape(-1, 1)
        self.restart(n)
        std_cache = np.zeros(10)
        self.g = 0
        while not self.should_terminate(problem, self.lambda_):
            Z = np.random.normal(0, 1, (n, self.lambda_))
            Y = np.dot(self.B, self.D * Z)
            X = self.m + (self.sigma * Y)
            f = np.array(problem(X.T))

            # select
            fidx = np.argsort(f)
            mu_best = fidx[: self.mu]
            
            # recombine
            m_old = self.m.copy()
            self.m = m_old + (1 * ((X[:, mu_best] - m_old) @ w).reshape(-1, 1))

            # adapt
            dm = (self.m - m_old) / self.sigma
            self.ps = (1 - cs) * self.ps + (np.sqrt(cs * (2 - cs) * mueff) * self.invC @ dm)
            self.sigma *= np.exp((cs / damps) * ((np.linalg.norm(self.ps) / chiN) - 1))
            hs = (
                np.linalg.norm(self.ps)
                / np.sqrt(1 - np.power(1 - cs, 2 * (problem.state.evaluations / self.lambda_)))
            ) < (1.4 + (2 / (n + 1))) * chiN

            dhs = (1 - hs) * cc * (2 - cc)
            self.pc = (1 - cc) * self.pc + (hs * np.sqrt(cc * (2 - cc) * mueff)) * dm

            if not self.old_lr:
                rank_one = c1 * self.pc * self.pc.T
                old_C = (1 - (c1 * dhs) - c1 - (cmu * w.sum())) * self.C
                rank_mu = cmu * (w * Y[:, mu_best] @ Y[:, mu_best].T)
            else:
                old_C = (1 - ccov) * self.C
                rank_one = (ccov / mueff) * self.pc * self.pc.T
                rank_mu = ccov * (1 - (1 / mueff)) * (w * Y[:, mu_best] @ Y[:, mu_best].T)
            self.C = old_C + rank_one + rank_mu
            
            self.g += 1
            std_cache[self.g % 10] = np.std(f)
            if (
                is_matrix_valid(self.C)
                or (not 1e-14 < self.sigma < 1e6)
            ):
                self.restart(n)
            else:
                self.C = np.triu(self.C) + np.triu(self.C, 1).T
                if not self.sep:
                    self.D, self.B = np.linalg.eigh(self.C)
                else:
                    self.D = np.diag(self.C)


            self.D = np.sqrt(self.D).reshape(-1, 1)
            self.invC = np.dot(self.B, self.D ** -1 * self.B.T)
            
            # print(problem.state.evaluations, self.sigma, np.mean(f), problem.state.current_best.y)
            
