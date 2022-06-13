from typing import Tuple

import ioh
import numpy as np

from .algorithm import Algorithm
from .plotting  import plot_positions_interactive

class ParticleSwarmOptimization(Algorithm):
    def __init__(
        self,
        s=10,
        max_iterations=1_000,
        inertia=0.5,
        psi_p=0.5,
        psi_g=0.5,
        verbose=False,
    ) -> None:
        self.s = s
        self.max_iterations = max_iterations
        self.inertia = inertia
        self.psi_p = psi_p
        self.psi_g = psi_g
        self.verbose = verbose

    def __call__(self, problem: ioh.problem.Real) -> Tuple[float, np.array]:
        positions = np.random.uniform(
            problem.constraint.lb,
            problem.constraint.ub,
            size=(self.s, problem.meta_data.n_variables),
        )
        domain = np.abs(
            np.array(problem.constraint.ub) - np.array(problem.constraint.lb)
        )
        velocity = np.random.uniform(
            -domain, domain, size=(self.s, problem.meta_data.n_variables)
        )

        f = np.array([problem(x) for x in positions])
        best_known_positions = positions.copy()
        best_known_values = f.copy()
        idx = np.argmin(f)
        ymin, xmin = f[idx], positions[idx, :]

        for i in range(self.max_iterations):
            rp = np.random.uniform(size=(self.s, 1))
            rg = np.random.uniform(size=(self.s, 1))
            velocity = (
                (self.inertia * velocity)
                + (self.psi_p * rp * (best_known_positions - positions))
                + (self.psi_g * rg * (xmin - positions))
            )

            positions += velocity
            f = np.array([problem(x) for x in positions])
            improvements = f < best_known_values
            best_known_positions[improvements] = positions[improvements]
            best_known_values[improvements] = f[improvements]

            idy = np.argmin(f)
            if f[idy] < ymin:
                ymin = f[idy]
                xmin = positions[idy]

            if self.verbose:
                plot_positions_interactive(positions)

        return ymin, xmin

