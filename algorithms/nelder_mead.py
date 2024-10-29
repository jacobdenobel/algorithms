from dataclasses import dataclass

import numpy as np
from scipy.stats.qmc import LatinHypercube, scale
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET


@dataclass
class NelderMead(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET * 20
    alpha: float = 1.0
    gamma: float = 2.0
    rho: float = 0.5
    sigma: float = 0.5
    min_shrinkage: float = 1e-9
    n_samples_per_dim: int = 2
    seed: int = 42

    def generate_simplex(self, problem: ioh.ProblemType):
        n_points = problem.meta_data.n_variables * self.n_samples_per_dim
        sampler = LatinHypercube(problem.meta_data.n_variables, seed=self.seed)
        sample = sampler.random(n=n_points)
        self.simplex = scale(sample, problem.bounds.lb, problem.bounds.ub)
        self.f_simplex = np.array(problem(self.simplex))

    def sort_simplex(self):
        f_idx = np.argsort(self.f_simplex)
        self.f_simplex = self.f_simplex[f_idx]
        self.simplex = self.simplex[f_idx]

    def update_worst(self, x: np.ndarray, f: float):
        self.f_simplex[-1] = f
        self.simplex[-1] = x.copy()

    def clip_step(self, dx):
        mask = np.abs(dx) < self.min_shrinkage
        dx[mask] = self.min_shrinkage * np.sign(dx[mask])
        return dx

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        self.generate_simplex(problem)

        while not self.should_terminate(problem, 1):
            self.sort_simplex()
            # print(self.f_simplex)
            # breakpoint()

            x0 = np.mean(self.simplex[:-1], axis=0)
            xr = x0 + (self.alpha * self.clip_step(self.simplex[0] - self.simplex[-1]))
            fr = problem(xr)

            # Reflection
            if self.f_simplex[0] <= fr and fr < self.f_simplex[-2]:
                # print("reflection")
                self.update_worst(xr, fr)
                continue

            # Expansion
            if fr < self.f_simplex[0]:
                # print("expansion")
                xe = x0 + (self.gamma * self.clip_step(xr - x0))
                fe = problem(xe)
                if fe < fr:
                    self.update_worst(xe, fe)
                else:
                    self.update_worst(xr, fr)
                continue

            # Contraction
            if self.f_simplex[-1] < fr:
                xr = self.simplex[-1].copy()
                fr = self.f_simplex[-1]

            xc = x0 + (self.rho * self.clip_step(xr - x0))
            fc = problem(xc)
            if fc < fr:
                # print("contraction")
                self.update_worst(xc, fc)
                continue

            # Shrink
            # print("shrink")
            self.simplex[1:] = self.simplex[0].copy() + (
                self.sigma * self.clip_step(self.simplex[1:] - self.simplex[0])
            )
            self.f_simplex[1:] = problem(self.simplex[1:])
