"""
The controller and experiment runner for genetic optimizer.
"""
import sys
sys.path.append("./")
import numpy as np
from typing import Dict
from core.ga.genetic_optimizer import GeneticOptimizer
from matplotlib import pyplot as plt

def obj_func(param: Dict[str, object]) -> float:
    x, y = param.values()
    f = (x+3)**2 + (y-2)**2 + 6
    return f


# Searching range
lb = 2e30
ub = -2e30
init_size = 500
epochs = 1000

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
    shot_prob=0.05,
    mutate_prob=0.05,
    verbose=False
)
optimizer.evaluation()

for e in range(epochs):
    print(f"Generation: [{e}/{epochs}]")
    optimizer.select()
    # print(optimizer.count_population())
    optimizer.evolve()
    optimizer.evaluation()

optimizer.count_population()
sum(isinstance(x, tuple) for x in optimizer.population)

print(f"Optimizer x-star found at {optimizer.population[0][0]}")
print(f"extremal value attained: {obj_func(optimizer.population[0][0]):0.5f}")

# print("More attentions are required if the maximizer/minimizer is near boundary.")

