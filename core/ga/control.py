"""
The controller and experiment runner for genetic optimizer.
"""
import sys
from typing import Dict

import numpy as np
# from matplotlib import pyplot as plt

sys.path.append("./")
from core.ga.genetic_optimizer import GeneticOptimizer
from core.ga.genetic_hpt import GeneticHPT

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

# Sample chromosomes.
f1 = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}
f2 = {"a": [7.0, 8.0, 9.0], "b": [10.0, 11.0, 12.0]}
f3 = {"a": [13.0, 14.0], "b": [15.0, 16.0]}

i1 = {"a": [1, 2, 3], "b": [4, 5, 6]}
i2 = {"a": [7, 8, 9], "b": [10, 11, 12]}
i3 = {"a": [13, 14], "b": [15, 16]}

optimizer = GeneticHPT(
    gene_pool=gene_pool,
    pop_size=init_size,
    eval_func=obj_func,
    mode="min",
    retain=0.5,
    shot_prob=0.05,
    mutate_prob=0.05,
    verbose=False
)

(a, b) = optimizer.cross_over(f1, i3)
print(a)
print(b)

optimizer.mutate(i2, mutate_prob=1.0)

optimizer.evaluate()
for e in range(epochs):
    print(f"Generation: [{e}/{epochs}]")
    optimizer.select()
    # print(optimizer.count_population())
    optimizer.evolve()
    optimizer.evaluate()

optimizer.count_population()
sum(isinstance(x, tuple) for x in optimizer.population)

print(f"Optimizer x-star found at {optimizer.population[0][0]}")
print(f"extremal value attained: {obj_func(optimizer.population[0][0]):0.5f}")

# print("More attentions are required if the maximizer/minimizer is near boundary.")
