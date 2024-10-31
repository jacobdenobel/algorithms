from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET
from .utils import is_matrix_valid


class M:
    def __get__(self, obj, objt=None):
        return obj.M

    def __set__(self, obj, value):
        obj.M = value


class OnePlusOneCMABase(Algorithm):

    @abstractmethod
    def update_strategy(self):
        pass

    def initialize(self, x0: np.ndarray, sigma0: float):
        self.x0 = np.asarray(x0).copy().reshape(-1, 1)
        self.sigma0 = sigma0

        self.sigma = sigma0
        self.n = len(x0)
        self.m = self.x0.copy()
        self.pc = np.zeros((self.n, 1))
        self.A = np.eye(self.n)

        self.M = np.eye(self.n)

        self.p_tgt_succ = 2 / 11
        self.damp = 1 + (self.n / 2)
        self.cp = 1 / 12
        self.cc = 2 / (self.n + 2)
        self.cc2 = self.cc * (2 - self.cc)
        self.ccov = 2 / (self.n**2 + 6)
        self.p_thres = 0.44
        self.p_succ = self.p_tgt_succ

    def mutate(self, problem: ioh.ProblemType):
        self.z = np.random.normal(size=(self.n, 1))
        self.y = self.A @ self.z
        self.x = self.m + (self.sigma * self.y)
        self.f = problem(self.x.ravel())
        self.has_improved = self.f < self.f_parent

    def adapt(self):
        self.p_succ = (1 - self.cp) * self.p_succ + self.cp * self.has_improved
        self.sigma = self.sigma * np.exp(
            (1 / self.damp) * ((self.p_succ - self.p_tgt_succ) / (1 - self.p_tgt_succ))
        )

        if self.f <= self.f_parent:
            self.m = self.x
            self.f_parent = self.f
            self.update_strategy()

    def restart(self, problem):
        sigma0 = (problem.bounds.ub[0] - problem.bounds.lb[0]) / 4
        x0 = np.random.uniform(problem.bounds.lb, problem.bounds.ub)
        self.initialize(x0, sigma0)
        self.f_parent = problem(self.m.ravel())

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        self.restart(problem)

        while not self.should_terminate(problem, 1):
            self.mutate(problem)
            self.adapt()
            if is_matrix_valid(self.A) or (not 1e-16 < self.sigma < 1e6):
                self.restart(problem)


@dataclass
class OnePlusOneCMAES(OnePlusOneCMABase):
    budget: int = DEFAULT_MAX_BUDGET

    C = M()

    def update_strategy(self):
        if self.p_succ < self.p_thres:
            self.pc = ((1 - self.cc) * self.pc) + (np.sqrt(self.cc2) * self.y)
            self.C = (1 - self.ccov) * self.C + (self.ccov * np.dot(self.pc, self.pc.T))
        else:
            self.pc = (1 - self.cc) * self.pc
            self.C = ((1 - self.ccov) * self.C) + (
                self.ccov * (np.dot(self.pc, self.pc.T) + (self.cc2 * self.C))
            )
        self.A = np.linalg.cholesky(self.C)
        # assert np.isclose(self.A@self.A.T, self.C).all()


@dataclass
class OnePlusOneCholeskyCMAES(OnePlusOneCMABase):
    budget: int = DEFAULT_MAX_BUDGET

    A_inv = M()

    def update_strategy(self):
        if self.p_succ < self.p_thres:
            self.pc = ((1 - self.cc) * self.pc) + (np.sqrt(self.cc2) * self.y)
            alpha = 1 - self.ccov
        else:
            self.pc = (1 - self.cc) * self.pc
            alpha = (1 - self.ccov) + (self.ccov * self.cc2)

        w = np.dot(self.A_inv, self.pc)
        sqrt_a = np.sqrt(alpha)
        w_norm = (w * w).sum()
        b_a = self.ccov / alpha

        self.A = sqrt_a * self.A + (
            (sqrt_a / w_norm) * (np.sqrt(1 + (b_a * w_norm)) - 1) * np.dot(self.pc, w.T)
        )
        self.A_inv = (1 / sqrt_a) * self.A_inv - (
            (1 / (sqrt_a * w_norm))
            * (
                (1 - (1 / np.sqrt(1 + (b_a * w_norm))))
                * np.dot(w, np.dot(w.T, self.A_inv))
            )
        )
