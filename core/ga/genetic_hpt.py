"""
Created: Dec. 26 2018
The genetic hyper parameter tunner for neural networks.
"""
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.extend("./")
from core.ga.genetic_optimizer import GeneticOptimizer

class GeneticHPT(GeneticOptimizer):
    """
    Genetic Hyper-Parameter Tuner
    """

    def __init__(self):
        raise NotImplementedError()

    def cross_over(
        p1: Dict[str, object],
        p2: Dict[str, object],
        self
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

        for k in p1.keys():
            if isinstance(p1[k], str):
                new_gene1 = np.random.choice([p1[k], p2[k]])
                new_gene2 = np.random.choice([p1[k], p2[k]])
            elif isinstance(p1[k], float):
                z = np.random.random()
                # Take the weighted average with random weight.
                new_gene1 = z * p1[k] + (1 - z) * p2[k]
                new_gene2 = (1 - z) * p1[k] + z * p2[k]
            elif isinstance(p1[k], int):
                z = np.random.random()
                # Take the weighted average with random weight.
                # And then round to the nearest integer.
                new_gene1 = int(np.round(
                    z * p1[k] + (1 - z) * p2[k]
                ))
                new_gene2 = int(np.round(
                    (1 - z) * p1[k] + z * p2[k]
                ))
            elif isinstance(p1[k], list):
                # TODO: add cross over rules for list with possibly different length.
                raise NotImplementedError()
            else:
                raise TypeError("Unsupported data type to cross over.")
            child1[k], child2[k] = new_gene1, new_gene2

        return (child1, child2)
