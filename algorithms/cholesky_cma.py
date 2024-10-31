from dataclasses import dataclass

import numpy as np
import ioh

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET
from .utils import is_matrix_valid

@dataclass
class CholeskyCMAES(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    
    def initialize(self, x0: np.ndarray, sigma0: float, lamb: int = None):
        self.x0 = np.asarray(x0).copy().reshape(-1, 1)
        self.sigma0 = sigma0
        self.sigma = sigma0
                
        self.n = len(x0)
        self.m = self.x0.copy()
        self.A = np.eye(self.n)
        self.A_inv = np.eye(self.n)
        self.ps = np.zeros((self.n, 1))
        self.pc = np.zeros((self.n, 1))
        
        # computed parameters
        self.lamb = lamb or int(4 + np.floor(3 * np.log(self.n)))
        self.mu = max(1, int(np.floor(self.lamb) / 2))
        
        # Weights
        log_mu = np.log(self.mu + 1)
        log_i = np.log(np.arange(1, self.mu + 1))
        w = (log_mu - log_i) / (self.mu * log_mu - np.sum(log_i))
        self.w = (w / w.sum()).reshape(-1, 1)
        self.mueff = 1.0/np.sum(np.square(self.w))

        # Learning rates
        self.cs = np.sqrt(self.mueff)/(np.sqrt(self.n) + np.sqrt(self.mueff))
        self.cc = 4.0 / (self.n + 4.0)
        self.ccov = 2.0 / np.square(self.n + np.sqrt(2.0))
        
        # Normalizers
        self.damps = 1.0 + 2.0*np.maximum(0.0, np.sqrt((self.mueff - 1.0)/(self.n + 1.0)) - 1.0) + self.cs
        self.chiN =  np.sqrt(self.n)*(1.0 - 1.0/(4.0*self.n) + 1.0/(21.0*np.square(self.n)))
        
        # print("cs", self.cs)
        # print("cc", self.cc)
        # print("ccov", self.ccov)
        # print("damps", self.damps)
                
    def mutate(self, problem):
        self.Z = np.random.normal(0, 1, size=(self.n, self.lamb))
        self.Y = np.dot(self.A, self.Z)
        self.X = self.m + self.sigma * self.Y
        self.f = np.array(problem(self.X.T))
        self.idx = np.argsort(self.f)[:self.mu]
    
    def adapt(self):
        self.zw   = self.Z[:,  self.idx].dot(self.w).reshape(-1, 1)
        self.m = self.X[:,  self.idx].dot(self.w).reshape(-1, 1)
        
        mucc      = np.sqrt(self.cc * (2 - self.cc) * self.mueff)
        self.pc   = (1 - self.cc) * self.pc + mucc * np.dot(self.A, self.zw)
        self.v    = np.dot(self.A_inv, self.pc)
        
        vnorm = (self.v * self.v).sum()
        sqrt1cc = np.sqrt(1.0 - self.ccov)
        sqrt1ccv = np.sqrt(1.0 + (self.ccov / (1.0 - self.ccov) * vnorm))
        va =  np.dot(self.v, np.dot(self.v.T, self.A_inv))
                
        self.A_inv = (1 / sqrt1cc * self.A_inv) - (
            (1 / (sqrt1cc * vnorm)) * (1 - (1 / sqrt1ccv)) * va
        )
        self.A = sqrt1cc * self.A + \
            ((sqrt1cc / vnorm * (sqrt1ccv - 1)) * self.pc * self.v.T)
            
        self.ps = (1 - self.cs) * self.ps + \
            (np.sqrt(self.cs * (2 - self.cs) * self.mueff) * self.zw)
        
        self.sigma = self.sigma * np.exp((self.cs / self.damps) * \
            ((np.linalg.norm(self.ps) / self.chiN)-1))

        
    def restart(self, problem, lamb: int = None):    
        sigma0 = (problem.bounds.ub[0] - problem.bounds.lb[0]) / 4
        x0 = np.random.uniform(problem.bounds.lb, problem.bounds.ub)
        
        self.initialize(x0, sigma0, lamb)
    
    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        self.restart(problem)
        
        while not self.should_terminate(problem, self.lamb):
            self.mutate(problem)
            self.adapt()
            if (
                is_matrix_valid(self.A)
                or is_matrix_valid(self.A_inv)
                or (not 1e-14 < self.sigma < 1e6)
            ):
                self.restart(problem)
            


