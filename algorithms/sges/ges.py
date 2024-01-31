from dataclasses import dataclass

import numpy as np
import ioh

from ..algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET
   
                

@dataclass
class GuidedES(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    lr: float = 0.2
    sigma: float = 0.01
    lambda_: int = 1
    k: int = 5

    def compute_gradients(self, x, problem, U, alpha=0.5):
        # Globalspace param
        a = self.sigma * np.sqrt(alpha / x.shape[0])
        # Subspace param
        c = self.sigma * np.sqrt((1 - alpha) / self.k)
        noise = a * np.random.normal(size=(self.lambda_, x.size))
        
        if alpha <= 0.5:
            noise += c * np.random.normal(size=(self.lambda_, self.k)) @ U.T

        grad = (np.array(problem(x + noise)) - np.array(problem(x - noise))).reshape(-1, 1)
        g_hat = (noise * grad).sum(axis=0) / (2 * self.lambda_ * self.sigma**2)
        return g_hat 

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        n = problem.meta_data.n_variables
        self.k = self.k or n // 2  # set default to half
        assert self.k <= n, f"subspace k({self.k}) should be smaller than n({n})"
        x_prime = np.zeros(n)
        f_prime = problem(x_prime)
        U, surg_grads = None, []
        try:
            while not self.should_terminate(problem, self.lambda_ * 2):
                if len(surg_grads) < self.k:
                    g_hat = self.compute_gradients(x_prime, problem, U, alpha=1)
                    surg_grads.append(g_hat)
                else:
                    U, _ = np.linalg.qr(np.array(surg_grads).T)
                    g_hat = self.compute_gradients(x_prime, problem, U, alpha=0.5)
                    surg_grads.pop(0)
                    surg_grads.append(g_hat)
                
                x_prime -= self.lr * g_hat
                f_prime = problem(x_prime)

        except KeyboardInterrupt:
            pass

        return x_prime, f_prime
    