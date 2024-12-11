# WIP: does not work properly yet!

from dataclasses import dataclass
import time

import numpy as np
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET


@dataclass
class LMMAES(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    lambda_: int = None
    mu: float = None
    m: int = None
    sigma0: float = 2
    verbose: bool = False

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        n = problem.meta_data.n_variables
        self.lambda_ = self.lambda_ or (4 + np.floor(3 * np.log(n))).astype(int)
        self.mu = self.lambda_ // 2

        echi = np.sqrt(n) * (1 - (1 / n / 4) - (1 / n / n / 21))
        wi_raw = np.log(self.lambda_ / 2 + 0.5) - np.log(np.arange(1, self.mu + 1))
        w = wi_raw / np.sum(wi_raw)
        mueff = 1 / np.sum(np.power(w, 2))
        
        self.m = self.m or int(4 + np.floor(3 * np.log(n)))
        cs = (2 * self.lambda_) / max(n, 7)

        cd = np.array([1 / (pow(1.5, i) * n) for i in range(self.m)]).reshape(-1, 1)
        cc = np.array([self.lambda_ /  (pow(4, i) * max(7, n)) for i in range(self.m)]).reshape(-1, 1)
        cc_sqrt = np.sqrt(mueff * cc * (2 - cc))
       
        sqrt_s = np.sqrt(cs * (2 - cs) * mueff)
        d_s = 1 + cs + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1)
        echi = np.sqrt(n)
        
        x_prime = np.random.uniform(problem.bounds.lb, problem.bounds.ub).reshape(-1, 1)
        sigma = self.sigma0
        ps = np.ones((n, 1))
        t = 0
        M = np.zeros((self.m, n))

        while not self.should_terminate(problem, self.lambda_):
            Z = np.random.normal(0, 1, (n, self.lambda_))
            Y = Z.copy()
            
            for j in range(min(t, self.m)):
                Y = (((1-cd[j]) * Y.T) + (cd[j] * M[j] * (M[j] * Y.T))).T
                
            X = x_prime + (sigma * Y)
            f = np.array(problem(X.T))
            idx = np.argsort(f)          
            mu_best = idx[: self.mu]
            wz = np.sum(w * Z[:, mu_best], axis=1, keepdims=True)
            d = np.sum(w * Y[:, mu_best], axis=1, keepdims=True)
            x_prime = x_prime + (sigma * d)
            
            ps = ((1 - cs) * ps) + (sqrt_s * wz)
            M = ((1 - cc) * M) + (cc_sqrt * wz.T)
            
            sigma = sigma * np.exp(cs / d_s * (np.linalg.norm(ps) / echi - 1))
            t += 1
            
            if self.verbose:
                print(problem.state.evaluations, sigma, np.mean(f), problem.state.current_best.y)
                time.sleep(.1)