"""As implemented from the paper: https://arxiv.org/abs/1806.10230"""
from dataclasses import dataclass

import numpy as np
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET


@dataclass
class GuidedES(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    lambda_: int = 4
    sigma: float = 0.01
    alpha: float = 0.5
    beta: float = 1.0
    lr: float = 0.2
    k: int = None

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        n = problem.meta_data.n_variables
        self.k = self.k or n // 2  # set default to half
        assert self.k <= n, f"subspace k({self.k}) should be smaller than n({n})"

        x_prime = np.random.uniform(size=n)
        f_prime = np.inf

        C = np.eye(n)
        surrogate_gradient = np.ones((n, self.k))

        t = 0
        s = 0
        cs = 0.3
        ds = 2 - (2 / n)
        try:
            f_old = None
            while not self.should_terminate(problem, self.lambda_ * 2):
                U, _ = np.linalg.qr(surrogate_gradient)
                # M = (self.alpha / n * C) + ((1 - self.alpha) / self.k) * U @ U.T
                # X = np.random.multivariate_normal(
                #     np.zeros(n), pow(self.sigma, 2) * M, size=self.lambda_
                # )

                c1 = self.sigma * np.sqrt(self.alpha / n)
                c2 = self.sigma * np.sqrt((1 - self.alpha) / self.k)

                z1 = np.random.normal(size=(n, self.lambda_))
                z2 = np.random.normal(size=(self.k, self.lambda_))
                X = ((c1 * z1) + (c2 * U.dot(z2)))

                f_pos = np.array(problem(x_prime + X.T))
                f_neg = np.array(problem(x_prime - X.T))
                scale = self.beta / (2 * pow(self.sigma, 2) * self.lambda_)
                grad = scale * np.sum((f_pos - f_neg) * X, axis=1)

                surrogate_gradient[:, t % self.k] = grad
                x_prime -= self.lr * grad
                f_prime = problem(x_prime)

                # f_new = np.minimum(f_pos, f_neg)
                # if f_old is not None:
                #     k_succ = (f_new < np.median(f_old)).sum()
                #     z = (2 / self.lambda_) * (k_succ - ((self.lambda_ + 1) / 2))
                #     s = ((1 - cs) * s) + (cs * z)
                #     self.sigma *= np.exp(s / ds)

                # f_old = f_new.copy()
                t += 1

        except KeyboardInterrupt:
            pass

        return x_prime, f_prime



@dataclass
class GuidedESV2(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    lambda_: int = 4
    mu: int = None
    sigma: float = 0.01
    alpha: float = 0.5
    beta: float = 1.0
    lr: float = 0.2
    k: int = None

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        n = problem.meta_data.n_variables
        self.k = self.k or n // 2  # Set default to half
        self.mu = self.mu or self.lambda_ // 2
        assert self.k <= n, f"subspace k({self.k}) should be smaller than n({n})"

        x_prime = np.random.uniform(size=n)
        m = x_prime.copy()
        f_prime = np.inf

        surrogate_gradient = np.ones((n, self.k))

        t = 0
        # s = 0
        # cs = 0.3
        # ds = 2 - (2 / n)
        echi = np.sqrt(n) * (1 - (1 / n / 4) - (1 / n / n / 21))
        
        M = np.eye(n)
        ps = np.ones((n, 1))
        f = np.empty(self.lambda_)
        try:
            while not self.should_terminate(problem, self.lambda_ + 1):
                U, _ = np.linalg.qr(surrogate_gradient)
                
                c1 = self.sigma * np.sqrt(self.alpha / n)
                c2 = self.sigma * np.sqrt((1 - self.alpha) / self.k)

                z1 = np.random.normal(size=(n, self.lambda_))
                z2 = np.random.normal(size=(self.k, self.lambda_))
                
                y1 = M.dot(z1)
                X = ((c1 * y1) + (c2 * U.dot(z2)))

                f_neg = np.array(problem(x_prime - X.T))
                f_pos = np.array(problem(x_prime + X.T))
                
                scale = self.beta / (2 * pow(self.sigma, 2) * self.lambda_)
                delta = (f_pos - f_neg)
                grad = scale * np.sum(delta * X, axis=1)

                surrogate_gradient[:, t % self.k] = grad
                x_prime -= self.lr * grad
                f_prime = problem(x_prime)
                t += 1

                neg_better = f_neg < f_pos
                f[neg_better] = f_neg[neg_better]
                f[~neg_better] = f_pos[~neg_better]
                fidx = np.argsort(f)
                mu_best = fidx[:self.mu]
                
                s_w = np.sign(delta[mu_best])
                w_raw = abs(delta[mu_best]) + 1e-16
                w = w_raw / np.sum(w_raw)

                mueff = 1 / np.sum(np.power(w, 2))
                c_s = (mueff + 2) / (n + mueff + 5)
                sqrt_s = np.sqrt(c_s * (2 - c_s) * mueff)
                d_s = 1 + c_s + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1)

                s_f = (1 - (2 * neg_better[mu_best]))
                

                Z = z1[:, mu_best] * s_f
                Y = y1[:, mu_best] * s_f
                z_prime = np.sum(
                    s_w * w * Z,
                    axis=1, 
                    keepdims=True
                )
                y_prime = np.sum(
                    s_w * w * Y,
                    axis=1, 
                    keepdims=True
                ) 

                ps = ((1 - c_s) * ps) + (sqrt_s * z_prime)
                c_1 = 2 / (pow(n + 1.3, 2) + mueff)
                c_mu = min(1 - c_1, 2 * (mueff - 2 + (1 / mueff)) / (pow(n + 2, 2) + mueff))

                M = (
                    ((1 - 0.5 * c_1 - 0.5 * c_mu) * M)
                    + ((0.5 * c_1) * M.dot(ps).dot(ps.T))
                    + ((0.5 * c_mu * w) * Y).dot(Z.T)
                )
                # print(M)
                # print(x_prime, f_prime)
                # breakpoint()
                # x_prime += xy_prime

                # print(m.ravel())
                # print(x_prime.ravel())               
                # m = m + (sigma * d)


                # self.sigma *= np.exp(c_s / d_s * (np.linalg.norm(ps) / echi - 1))
                # breakpoint()

                # idx = np.argsort()
                # mu_best = idx[: self.mu]
               

        except KeyboardInterrupt:
            pass

        return x_prime, f_prime