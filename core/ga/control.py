"""
The controller and experiment runner for genetic optimizer.
"""
import numpy as np
from typing import Dict
from core.ga.genetic import GeneticOptimizer
from matplotlib import pyplot as plt

def obj_func(
    param: Dict[str, object]
) -> float:
    x = param["x"]
    return 2 * x ** 8 + 3 * x + 4

# Searching range
lb = -10.0
ub = 10.0
init_size = 500
epochs = 20

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
    print(optimizer.count_population())
    optimizer.evolve()

ranked_minimizers = [entity["x"] for entity in optimizer.population]
ranked_values = [obj_func(entity) for entity in optimizer.population]
# plt.plot(ranked_values)
# plt.show()

# plt.plot(ranked_values)
# plt.show()

print(f"Optimizer x-star found at {optimizer.population[0]['x']:0.7f} \
extremal value attained at {obj_func(optimizer.population[0]):0.7f}")
