"""
The controller and experiment runner for genetic optimizer.
"""
import sys
sys.path.append("./")
import numpy as np
from typing import Dict
from core.ga.genetic import GeneticOptimizer
from matplotlib import pyplot as plt

def obj_func(param: Dict[str, object]) -> float:
    x, y = param.values()
    f = (x+3)**2 + (y-2)**2 + 6
    return f


# Searching range
lb = 20
ub = -20
init_size = 1000
epochs = 500

candidates = (ub - lb) * np.random.random(init_size) + lb

gene_pool = {
    "x": list(candidates),
    "y": list(candidates)
    }

optimizer = GeneticOptimizer(
    gene_pool=gene_pool,
    pop_size=init_size,
    eval_func=obj_func,
    mode="min",
    retain=0.5,
    verbose=False
)

for e in range(epochs):
    print(f"Generation: [{e}/{epochs}]")
    optimizer.evaluation()
    optimizer.select()
    # print(optimizer.count_population())
    optimizer.evolve()


print(f"Optimizer x-star found at {optimizer.population[0][0]}")
print(f"extremal value attained: {obj_func(optimizer.population[0][0]):0.5f}")

# print("More attentions are required if the maximizer/minimizer is near boundary.")

