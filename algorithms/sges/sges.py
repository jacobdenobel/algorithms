from dataclasses import dataclass

import numpy as np
import ioh

from ..algorithm import Algorithm, DEFAULT_MAX_BUDGET, SolutionType


@dataclass
class SelfGuidedES(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    lr: float = 0.01
    sigma: float = 0.01
    lambda_: int = 10
    k: int = 1
    auto_alpha: bool = True

    def compute_gradients(self, x, problem, U=None, alpha=0.2):
        grad = 0
        grad_loss, random_loss = [], []
        for i in range(self.lambda_):
            if abs(np.random.uniform()) < alpha:
                noise = self.sigma / np.sqrt(self.k) * np.random.randn(1, self.k) @ U.T
                noise = noise.reshape(x.shape)
                pos_loss, neg_loss = problem(x + noise), problem(x - noise)
                grad_loss.append(min(pos_loss, neg_loss))
            else:
                noise = self.sigma / np.sqrt(len(x)) * np.random.randn(1, len(x))
                noise = noise.reshape(x.shape)
                pos_loss, neg_loss = problem(x + noise), problem(x - noise)
                random_loss.append(min(pos_loss, neg_loss))
            grad += noise * (pos_loss - neg_loss)

        g_hat = grad / (2 * self.lambda_ * self.sigma**2)

        mean_grad_loss = 10000 if len(grad_loss) == 0 else np.mean(np.asarray(grad_loss))
        mean_random_loss = (
            10000 if len(random_loss) == 0 else np.mean(np.asarray(random_loss))
        )

        return g_hat, mean_grad_loss, mean_random_loss

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        n = problem.meta_data.n_variables
        self.k = self.k or n // 2  # set default to half
        assert self.k <= n, f"subspace k({self.k}) should be smaller than n({n})"

        # x_prime = np.zeros(n)
        x_prime = np.random.normal(size=n)
        f_prime = problem(x_prime)
        alpha, U, surg_grads = 0.5, None, []

        try:
            while not self.should_terminate(problem, self.lambda_):
                if len(surg_grads) < self.k:
                    g_hat, *_ = self.compute_gradients(x_prime, problem, U, 0)
                    surg_grads.append(g_hat)
                else:
                    U, _ = np.linalg.qr(np.array(surg_grads).T)
                    (
                        g_hat,
                        mean_grad_loss,
                        mean_random_loss,
                    ) = self.compute_gradients(x_prime, problem, U, alpha)
                    if self.auto_alpha:
                        alpha = np.clip(
                            alpha * 1.005
                            if mean_grad_loss < mean_random_loss
                            else alpha / 1.005
                        , 0.3, 0.7)

                    surg_grads.pop(0)
                    surg_grads.append(g_hat)

                x_prime -= self.lr * g_hat
                f_prime = problem(x_prime)

        except KeyboardInterrupt:
            pass

        return x_prime, f_prime
