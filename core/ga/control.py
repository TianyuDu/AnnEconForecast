"""
The controller and experiment runner for genetic optimizer.
"""
import numpy as np
from typing import Dict
from core.ga.genetic import GeneticOptimizer as GGO

def obj_func(
    param: Dict[str, object]
) -> float:
    x = param["x"]
    return 2 * x ** 2 + 3 * x + 4