from dataclasses import dataclass

import numpy as np
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET


@dataclass
class ARSV1(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    sigma0: float = 0.02     # learning rate alpha
    lambda_: int = 16        # n offspring for each direction
    mu: int = 16             # best offspring
    eta: float = 0.03        # noise parameter  
    
    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        n = problem.meta_data.n_variables
        x_prime = np.zeros((n, 1))
        try:
            while not self.should_terminate(problem, self.lambda_):
                delta = np.random.normal(size=(n, self.lambda_))

                neg = x_prime - (self.eta * delta)
                pos = x_prime + (self.eta * delta)

                neg_reward = -np.array(problem(neg.T))
                pos_reward = -np.array(problem(pos.T))

                best_rewards = np.maximum(neg_reward, pos_reward)
                idx = np.argsort(best_rewards)[::-1]

                f = np.r_[neg_reward, pos_reward]
                sigma_rewards = f.std() + 1e-12
                weight = self.sigma0 / (self.lambda_ * sigma_rewards)

                delta_rewards = pos_reward - neg_reward
                x_prime += (weight * (delta_rewards[idx] * delta[:, idx]).sum(axis=1, keepdims=True))

        except KeyboardInterrupt:
            pass
        return x_prime