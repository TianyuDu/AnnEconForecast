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
        retain: float=0.3,
        shot_prob: float=0.05,
        mutate_prob: float=0.05,
        verbose: bool=False
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
            """
            Helper function, crossover float objects.
            """
            # Take the random convex combination.
            z = np.random.random()
            new_f1 = z * f1 + (1 - z) * f2
            new_f2 = (1 - z) * f1 + z * f2
            assert all(isinstance(x, float) for x in [new_f1, new_f2]), "Wrong return type(s), both should be float."
            return new_f1, new_f2

        def mixup_integer(
            i1: int,
            i2: int,
            choose: bool=True
        ) -> (int, int):
            """
            Helper function, crossover integer objects.
            Args:
                choose: randomly choose an integer or taking the convex combination.
            """
            if choose:
                # TODO: Consider which method to use.
                # Randomly distribute.
                new_i1, new_i2 = np.random.choice([i1, i2], size=2, replace=False)
            else:
                f1, f2 = mixup_float(float(i1), float(i2))
                rd = lambda x: int(np.round(x))
                new_i1, new_i2 = rd(f1), rd(f2)
            assert all(isinstance(x, int) for x in [new_i1, new_i2]), "Wrong return type(s), both should be int."
            return new_i1, new_i2

        def mixup_numerical(
            x1: Union[float, int],
            x2: Union[float, int]
        ) -> Tuple[Union[float, int]]:
            """
            Helper function, crossover numerical objects.
            """
            assert type(x1) == type(x2), "x1 and x2 should have the same numerical type."
            if isinstance(x1, int):
                return mixup_integer(x1, x2)
            elif isinstance(x1, float):
                return mixup_float(x1, x2)
            else:
                raise TypeError("Unsupported data type.")

        for k in p1.keys():
            if type(p1[k]) != type(p2[k]):
                # For different types to cross over,
                # E.g. p1 has grad clipping activated (10.0) and p2 has no gradient clipping (None)
                # Randomly distribute.
                new_gene1, new_gene2 = np.random.choice(
                    [p1[k], p2[k]],
                    size=2,
                    replace=False
                )
            elif isinstance(p1[k], str) or isinstance(p1[k], int):
                new_gene1 = np.random.choice([p1[k], p2[k]])
                new_gene2 = np.random.choice([p1[k], p2[k]])
            elif isinstance(p1[k], float):
                new_gene1, new_gene2 = mixup_float(p1[k], p2[k])
            elif isinstance(p1[k], list):
                if len(p1[k]) == len(p2[k]):
                    # Case 1: same length.
                    # e.g. two parameter sets give two multi-layer LSTM with neurons [16, 32, 64] and [32, 64, 128]
                    # The data type will be preserved after cross over.
                    mixed_gene = [
                        mixup_numerical(x1, x2)
                        for x1, x2 in zip(p1[k], p2[k])
                    ]
                else:
                    # Case 2: with different length.
            else:
                raise TypeError("Unsupported data type to cross over.")

            child1[k], child2[k] = new_gene1, new_gene2

        return (child1, child2)
