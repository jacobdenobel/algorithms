import abc
from typing import Tuple

import ioh
import numpy as np

SolutionType = Tuple[float, np.ndarray]

class Algorithm(abc.ABC):
    
    @abc.abstractmethod
    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        pass