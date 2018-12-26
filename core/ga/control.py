"""
The controller and experiment runner for genetic optimizer.
"""
import sys
sys.path.append("./")
import numpy as np
from typing import Dict
from core.ga.genetic import GeneticOptimizer
from matplotlib import pyplot as plt

def obj_func(
    param: Dict[str, object]
) -> float:
    x = param["x"]
    f = - (2*x**7 + 923123213*x**12 + 3*x + 21312)
    return f

def obj_func2(param):
    x, y, z = list(param.values())
    f = (x-3.6)**2 + (3*y)**2 + (z*2)**2 + 6
    return f

# Searching range
lb = 20
ub = -20
init_size = 1000
epochs = 500

candidates = (ub - lb) * np.random.random(init_size) + lb

gene_pool = {
    "x": list(candidates),
    "y": list(candidates),
    "z": list(candidates)
    }

optimizer = GeneticOptimizer(
    gene_pool=gene_pool,
    pop_size=init_size,
    eval_func=obj_func2,
    mode="min",
    retain=0.5,
    verbose=False
)

for e in range(epochs):
    print(f"Generation: [{e}/{epochs}]")
    optimizer.evaluation()
    optimizer.select(verbose=False)
    # print(optimizer.count_population())
    optimizer.evolve()

ranked_minimizers = [entity["x"] for entity in optimizer.population]
ranked_values = [obj_func2(entity) for entity in optimizer.population]
# plt.plot(ranked_values)
# plt.show()

# plt.plot(ranked_values)
# plt.show()

print(f"Optimizer x-star found at {optimizer.population[0]}")
print(f"extremal value attained: {obj_func2(optimizer.population[0]):0.5f}")

# print("More attentions are required if the maximizer/minimizer is near boundary.")

