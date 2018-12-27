"""
Created: Dec. 26 2018
The genetic hyper parameter tuner for neural networks.
"""
import sys
from typing import Dict, List, Tuple, Union, Callable

import numpy as np

sys.path.extend("./")
from core.ga.genetic_optimizer import GeneticOptimizer

class GeneticHPT(GeneticOptimizer):
    """
    Genetic Hyper-Parameter Tuner for neural networks.
    """

    def __init__(
        self,
        gene_pool: Dict[str, Union[object, List[object]]],
        pop_size: int,
        eval_func: Callable[[Dict[str, object]], Union[float, int]],
        mode: Union["min", "max"],
        retain: float = 0.3,
        shot_prob: float = 0.05,
        mutate_prob: float = 0.05,
        verbose: bool = False
    ) -> None:
        """
        Docstring refer to the parent class GeneticOptimizer.
        """
        super().__init__(
            gene_pool,
            pop_size,
            eval_func,
            mode,
            retain,
            shot_prob,
            mutate_prob,
            verbose
        )

    def cross_over(
        self,
        p1: Dict[str, object],
        p2: Dict[str, object]
    ) -> Tuple[dict]:
        """
        Individual cross over method. This method should only be called in
        evolve phase.
        If the feature is a string, randomly choose one from the chromosome of 
        parents.
        If the method is a float or integer, cross over methods take a weighted
        average with random weight. For integers, a round with int operation will 
        be added.
        """
        child1 = {key: None for key in p1.keys()}
        child2 = {key: None for key in p1.keys()}

        def mixup_float(f1: float, f2: float) -> (float, float):
            # Take the random convex combination.
            z = np.random.random()
            new_f1 = z * f1 + (1 - z) * f2
            new_f2 = (1 - z) * f1 + z * f2
            return new_f1, new_f2

        for k in p1.keys():
            if isinstance(p1[k], str) or isinstance(p1[k], int):
                new_gene1 = np.random.choice([p1[k], p2[k]])
                new_gene2 = np.random.choice([p1[k], p2[k]])
            elif isinstance(p1[k], float):
                new_gene1, new_gene2 = mixup_float(p1[k], p2[k])
            elif isinstance(p1[k], list):
                # TODO: add cross over rules for list with possibly different length.
                
            else:
                raise TypeError("Unsupported data type to cross over.")
            child1[k], child2[k] = new_gene1, new_gene2

        return (child1, child2)
