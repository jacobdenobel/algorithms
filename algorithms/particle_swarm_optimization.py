from dataclasses import dataclass

import ioh
import numpy as np

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET
from .plotting  import plot_positions_interactive

@dataclass
class ParticleSwarmOptimization(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    s: int = 50
    inertia: float = .5
    psi_p: float = 1.5
    psi_g: float = 1.5
    verbose: bool = False

    def __call__(self, problem: ioh.problem.RealSingleObjective) -> SolutionType:
        positions = np.random.uniform(
            problem.bounds.lb,
            problem.bounds.ub,
            size=(self.s, problem.meta_data.n_variables),
        )
        domain = np.abs(
            np.array(problem.bounds.ub) - np.array(problem.bounds.lb)
        )
        velocity = np.random.uniform(
            -domain, domain, size=(self.s, problem.meta_data.n_variables)
        )

        f = np.array([problem(x) for x in positions])
        best_known_positions = positions.copy()
        best_known_values = f.copy()
        idx = np.argmin(f)
        ymin, xmin = f[idx], positions[idx, :]

        while self.not_terminate(problem, self.s):
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

