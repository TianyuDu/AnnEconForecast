"""
The controller and experiment runner for genetic optimizer.
"""
import numpy as np
from typing import Dict
from core.ga.genetic import GeneticOptimizer

def obj_func(
    param: Dict[str, object]
) -> float:
    x = param["x"]
    return 2 * x ** 2 + 3 * x + 4

# Searching range
lb = -10
ub = 10
init_size = 100
epochs = 30

candidates = (ub - lb) * np.random.random(init_size) + lb

gene_pool = {"x": list(candidates)}

optimizer = GeneticOptimizer(
    gene_pool=gene_pool,
    pop_size=init_size,
    eval_func=obj_func,
    mode="min",
    retain=0.5,
    verbose=True
)

for _ in range(epochs):
    optimizer.evaluation()
    optimizer.select(verbose=True)
    optimizer.evolve()
