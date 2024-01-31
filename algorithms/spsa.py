from dataclasses import dataclass

import numpy as np
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET

@dataclass
class SPSA(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    a: float = 0.1
    c: float = 0.1
    alpha: float = 0.602
    gamma: float = 0.101


    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        n = problem.meta_data.n_variables
        x_prime = np.zeros(n)
        f_prime = problem(x_prime)

        A = self.budget * 0.1

        k = 1
        try:
            while not self.should_terminate(problem, 1):
                # Generate a random perturbation vector with elements that are either +1 or -1
                delta = np.random.choice([-1, 1], size=n)

                # ak = self.a / ((k + A) ** (self.alpha))
                # ck = self.c / (k ** (self.gamma))
                ck = self.c
                ak = self.a

                # Evaluate the objective function at the positive and negative perturbations
                loss_plus = problem(x_prime + ck * delta)
                loss_minus = problem(x_prime - ck * delta)

                # Calculate the gradient estimate using the perturbations
                gradient = (loss_plus - loss_minus) / (2 * ck * delta)

                # Update the parameter values
                x_prime = x_prime - ak * gradient

                # If the new parameter values result in a lower loss, update the best values
                f_prime = problem(x_prime)

                k += 1
        except KeyboardInterrupt:
            pass

        return x_prime, f_prime



# @dataclass
# class SPSA(Algorithm):
#     budget: int = DEFAULT_MAX_BUDGET
#     alpha: float = 0.602
#     gamma: float = 0.101

#     def grad(self, problem, w, ck):
#         # number of parameters
#         p = len(w)

#         # bernoulli-like distribution
#         deltak = np.random.choice([-1, 1], size=p)

#         # simultaneous perturbations
#         ck_deltak = ck * deltak

#         # gradient approximation
#         delta_l = problem(w + ck_deltak) - problem(w - ck_deltak)

#         return delta_l / (2 * ck_deltak)

#     def __call__(self, problem: ioh.ProblemType) -> SolutionType:
#         n = problem.meta_data.n_variables
#         x_prime = np.random.standard_normal(n)
#         f_prime = problem(x_prime)

#         c = 1e-2  # a small number

#         # A is <= 10% of the number of iterations
#         A = self.budget * 0.1

#         # order of magnitude of first gradients
#         magnitude_g0 = np.abs(self.grad(problem, x_prime, c).mean())

#         # the number 2 in the front is an estimative of
#         # the initial changes of the parameters,
#         # different changes might need other choices
#         a = 2 * ((A + 1) ** self.alpha) / magnitude_g0

#         k = 1
#         try:
#             while not self.should_terminate(problem, 1):
#                 # update ak and ck
#                 ak = a / ((k + A) ** (self.alpha))
#                 ck = c / (k ** (self.gamma))

#                 # estimate gradient
#                 gk = self.grad(problem, x_prime, ck)
#                 # breakpoint()
#                 # update parameters
#                 x_prime -= ak * gk
#                 f_prime = problem(x_prime)
#                 print(f_prime)
                
#                 import time
#                 time.sleep(.1)
#                 k += 1
#         except KeyboardInterrupt:
#             pass

#         return x_prime, f_prime
